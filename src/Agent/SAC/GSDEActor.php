<?php
namespace Rindow\RL\Agents\Agent\SAC;

# ─────────────────────────────────────────────
# gSDE Actor
# ─────────────────────────────────────────────
#    PyTorch 版との対応:
#        nn.Sequential(Linear, ReLU, ...)  → tf.keras.Sequential([Dense(...), ...])
#        nn.Parameter(tensor)              → tf.Variable(..., trainable=True)
#        forward_inference(obs, W_noise)   → そのまま同名メソッド
#        forward_train(obs)                → そのまま同名メソッド
#        sample_noise()                    → そのまま同名メソッド
class GSDEActor extends AbstractModel
{
    private object $la;
    private object $g;
    private int $act_dim;
    private int $latents_dim;
    protected Model $phi_net;  // must be protected or public to be found by trainable variables
    protected Layer $mu_head;    // must be protected or public
    protected Variable $log_std; // must be protected of public
    
    public function __construct(
        Builder $nn,
        int $obs_dim, int $act_dim, int $latent_dim = GSDE_LATENT_DIM)
    {
        parent::__construct($nn);
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        
        $this->act_dim    = $act_dim;
        $this->latents_dim = $latent_dim;

        # 共有特徴抽出器  (PyTorch: phi_net)
        $this->phi_net = $nn->models->Sequential([
            $nn->layers->Dense(HIDDEN_DIM, activation:"relu",
                                  input_shape:[$obs_dim]),
            $nn->layers->Dense($latent_dim, activation:"relu"),
        ]);

        # 平均ヘッド  (PyTorch: mu_head = nn.Linear)
        $this->mu_head = $nn->layers->Dense($act_dim, input_shape:[$latent_dim]);

        # gSDE 対数標準偏差  (PyTorch: nn.Parameter)
        $this->log_std = $this->g->Variable(
            $this->la->fill(-1.0,$this->la->alloc([$act_dim, $latent_dim],dtype:NDArray::float32)),
            trainable:True, name:"log_std"
        );
    }

    # ── 共通特徴抽出 ────────────────────────────
    private function phi_and_mu(Variable $obs) : array
    {
        $phi = $this->phi_net->forward($obs);    # (B, latent_dim)
        $mu  = $this->mu_head->forward($phi);    # (B, act_dim)
        return [$phi, $mu];
    }

    private function std_W() : Variable
    {
        return $this->g->exp($this->log_std);   # (act_dim, latent_dim)
    }

    # ── ① ノイズサンプル ────────────────────────
    #    W_noise ~ N(0, std_W²) をサンプルして返す。
    #    ループ変数として保持し、GSDE_RESET_FREQ ごとに再呼び出し。
    #
    #    PyTorch: torch.randn_like(std) * std
    #    TF:      tf.random.normal(tf.shape(std)) * std
    public function sample_noise() : Variable
    {
        $g = $this->g;
        $std = $this->std_W();
        $eps = $g->randomNormal($std);
        return $g->mul($eps, $std);  # (act_dim, latent_dim)
    }

    # ── ② 推論パス（勾配なし） ──────────────────
    #   PyTorch: with torch.no_grad(): ...
    #   TF: tape 外から呼ぶことで自動的に勾配追跡なし
    public function forward_inference(Variable $obs, Variable $W_noise) : Variable
    {
        [$phi, $mu] = $this->phi_and_mu($obs);
        # (act_dim, latent_dim) @ (latent_dim, 1) → (act_dim, 1) → (1, act_dim)
        $phi_T = $this->g->transpose($phi);
        $matmul = $this->g->matmul($W_noise, $phi_T);
        $noise = $this->g->transpose($matmul); # (1, act_dim)
        return $this->g->tanh($this->g->add($mu, $noise));
    }

    # ── ③ 学習パス（GradientTape 内で呼ぶ） ─────
    # """
    # 外部状態に依存しない自己完結パス。
    # GradientTape スコープ内で呼ぶことで log_std への勾配が流れる。
    #
    # PyTorch の reparameterization:
    #     eps   = torch.randn(B, act_dim, latent_dim)
    #     W     = eps * std_W.unsqueeze(0)
    #     noise = torch.einsum("bl,bal->ba", phi, W)
    #
    # TF の reparameterization:
    #     eps   = tf.random.normal([B, act_dim, latent_dim])
    #     W     = eps * std_W[tf.newaxis, :, :]
    #     noise = tf.einsum("bl,bal->ba", phi, W)
    public function forward_train(Variable $obs) : array
    {
        $g = $this->g;
        [$phi, $mu] = $this->phi_and_mu($obs);
        $std_W   = $this->std_W();                      # (act_dim, latent_dim)

        $B     = $obs->shape()[0];
        $eps   = $g->randomNormal($std_W,batchShape:[$B]);
        $W     = $g->mul($eps, $std_W);  # (B, act_dim, latent_dim) eps <- broadcast $std_W
        
        $phi_reshaped = $g->reshape($phi, [$B, $this->latents_dim, 1]);
        $matmul = $g->matmul($W, $phi_reshaped);
        $noise = $g->squeeze($matmul, 2);         # (B, act_dim)

        $x_t = $g->add($mu, $noise);
        $y_t = $g->tanh($x_t);

        # sigma_z(s) = sqrt( std_W² @ phi² )
        # PyTorch: (std_W.pow(2) @ phi.T.pow(2)).sqrt().T
        # TF:      tf.transpose( tf.sqrt(std_W**2 @ tf.transpose(phi**2)) )
        $std_W_sq = $g->square($std_W);
        $phi_sq = $g->square($phi);
        $phi_sq_T = $g->transpose($phi_sq);
        $matmul_sq = $g->matmul($std_W_sq, $phi_sq_T);
        $sqrt = $g->sqrt($matmul_sq);
        $sigma_z = $g->transpose($sqrt);
        $sigma_z = $g->maximum($sigma_z,$g->constant(1e-6));

        $log_sigma = $g->log($sigma_z);
        $diff = $g->sub($x_t, $mu);
        $diff_sq = $g->square($diff);
        $sigma_z_sq = $g->square($sigma_z);
        $two_sigma_z_sq = $g->mul(2.0, $sigma_z_sq);
        $term3 = $g->div($diff_sq, $two_sigma_z_sq);
        
        $log_prob = $g->sub(-0.91893853320467, $log_sigma);
        $log_prob = $g->sub($log_prob, $term3);

        $y_t_sq = $g->square($y_t);
        $tanh_corr_inner = $g->add($g->sub(1.0, $y_t_sq), 1e-6); # tanh 補正
        $tanh_corr = $g->log($tanh_corr_inner);
        $log_prob = $g->sub($log_prob, $tanh_corr);
        
        $log_prob = $g->reduceSum($log_prob, axis: -1, keepdims: true);

        return [$y_t, $log_prob];
    }

    # tf.keras.Model の call は forward_train を使う
    public function call(Variable $obs) : array
    {
        return $this->forward_train($obs);
    }

}

