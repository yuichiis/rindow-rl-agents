<?php
# """
# MountainCarContinuous-v0  SAC + gSDE  (TensorFlow / GradientTape 版)
# =====================================================================
# 
# PyTorch 版からの対応表:
#     torch.nn.Module          → tf.keras.Model
#     nn.Parameter             → tf.Variable(trainable=True)
#     optimizer.zero_grad()
#     loss.backward()          → with tf.GradientTape() as tape:
#     optimizer.step()         →     grads = tape.gradient(loss, vars)
#                              →     opt.apply_gradients(zip(grads, vars))
#     torch.no_grad()          → tape 外で計算 + tf.stop_gradient()
#     torch.randn_like(x)      → tf.random.normal(tf.shape(x))
#     torch.einsum             → tf.einsum
#     F.mse_loss               → tf.reduce_mean((a - b) ** 2)
#     torch.min(a, b)          → tf.minimum(a, b)
#     tensor.clamp(min=v)      → tf.maximum(tensor, v)
# 
# 依存: pip install gymnasium tensorflow numpy
# """

//import numpy as np
//import tensorflow as tf
//import gymnasium as gym
//import random

require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Layer\Layer;

# ─────────────────────────────────────────────
# ハイパーパラメータ
# ─────────────────────────────────────────────
const ENV_ID          = "MountainCarContinuous-v0";
const SEED            = 42;
const TOTAL_STEPS     = 100000;
const START_STEPS     = 1000;
const BATCH_SIZE      = 256;
const BUFFER_SIZE     = 100000;
const LR_ACTOR        = 3e-4;
const LR_CRITIC       = 3e-4;
const LR_ALPHA        = 3e-4;
const GAMMA           = 0.99;
const TAU             = 0.005;
const HIDDEN_DIM      = 256;
const ALPHA_INIT      = 1.0;
const GSDE_LATENT_DIM = 64;
const GSDE_RESET_FREQ = 16;
const UPDATE_EVERY    = 1;
const EVAL_EVERY      = 1_000;
const EVAL_EPISODES   = 3;

//print(f"TensorFlow version: {tf.__version__}")
//print(f"GPUs: {tf.config.list_physical_devices('GPU')}")


# ─────────────────────────────────────────────
# リプレイバッファ  (numpy のまま、変更なし)
# ─────────────────────────────────────────────
class ReplayBuffer
{
    private Builder $nn;
    private object $la;
    private object $g;
    private int $capacity;
    private int $obs_dim;
    private int $act_dim;
    private int $ptr;
    private int $size;
    private NDArray $obs;
    private NDArray $rewards;
    private NDArray $next_obs;
    private NDArray $dones;
    private NDArray $actions;

    public function __construct(
        Builder $nn,
        int $capacity,
        int $obs_dim,
        int $act_dim
        )
    {
        $la = $nn->backend()->primaryLA();
        $this->nn = $nn;
        $this->la = $la;
        $this->g = $this->nn->gradient();
        $this->capacity = $capacity;
        $this->ptr = 0;
        $this->size = 0;
        $this->obs      = $la->zeros($la->alloc([$capacity, $obs_dim], dtype:NDArray::float32));
        $this->rewards  = $la->zeros($la->alloc([$capacity, 1],       dtype:NDArray::float32));
        $this->next_obs = $la->zeros($la->alloc([$capacity, $obs_dim], dtype:NDArray::float32));
        $this->dones    = $la->zeros($la->alloc([$capacity, 1],       dtype:NDArray::float32));
        $this->actions  = $la->zeros($la->alloc([$capacity, $act_dim], dtype:NDArray::float32));
    }

    public function add(
        NDArray $obs,
        NDArray $action,
        float $reward,
        NDArray $next_obs,
        bool $done
        ) : void
    {
        $this->obs[$this->ptr]      = $obs;
        $this->actions[$this->ptr]  = $action;
        $this->rewards[$this->ptr][0] = $reward;
        $this->next_obs[$this->ptr] = $next_obs;
        $this->dones[$this->ptr][0] = $done;
        $this->ptr  = ($this->ptr + 1) % $this->capacity;
        $this->size = min($this->size + 1, $this->capacity);
    }

    public function sample(int $batch_size) : array
    {
        $idx = $this->la->randomSequence($this->size, $batch_size);
        return [
            $this->la->gather($this->obs,$idx),
            $this->la->gather($this->actions,$idx),
            $this->la->gather($this->rewards,$idx),
            $this->la->gather($this->next_obs,$idx),
            $this->la->gather($this->dones,$idx),
        ];
    }
}

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

# ─────────────────────────────────────────────
# Critic (Double Q)
# ─────────────────────────────────────────────
class QNetwork extends AbstractModel
{
    private object $g;
    protected AbstractModel $model; // must be protected or public to be found by trainable variables

    public function __construct(Builder $nn, int $obs_dim, int $act_dim, int $hidden_dim)
    {
        parent::__construct($nn);
        $this->g = $nn->gradient();
        $this->model = $nn->models->Sequential([
            $nn->layers->Dense($hidden_dim, activation: 'relu', input_shape: [$obs_dim + $act_dim]),
            $nn->layers->Dense($hidden_dim, activation: 'relu'),
            $nn->layers->Dense(1),
        ]);
    }

    public function call(Variable $obs, Variable $action, ?bool $training=null) : Variable
    {
        $x = $this->g->concat([$obs, $action], axis: -1);
        return $this->model->forward($x, $training);
    }
}

#   PyTorch の Critic(q1, q2) に対応。
#   TF では Functional API で 2 つの独立したサブモデルを保持する。
class Critic extends AbstractModel
{
    public QNetwork $q1;
    public QNetwork $q2;

    public function __construct(Builder $nn, int $obs_dim, int $act_dim, int $hidden_dim)
    {
        parent::__construct($nn);
        $this->q1 = new QNetwork($nn, $obs_dim, $act_dim, $hidden_dim);
        $this->q2 = new QNetwork($nn, $obs_dim, $act_dim, $hidden_dim);
    }

    public function call(Variable $obs, Variable $action, ?bool $training=null) : array
    {
        return [$this->q1->forward($obs, $action, $training), $this->q2->forward($obs, $action, $training)];
    }
}

# ─────────────────────────────────────────────
# ソフトアップデートユーティリティ
# ─────────────────────────────────────────────
#   PyTorch:
#       for p, p_tgt in zip(src.parameters(), tgt.parameters()):
#           p_tgt.data.copy_(tau * p + (1-tau) * p_tgt)
#   TF:
#       source.weights / target.weights をペアで assign
function soft_update(object $g, AbstractModel $source, AbstractModel $target, float $tau) : void
{
    $src_vars = $source->trainableVariables();
    $tgt_vars = $target->trainableVariables();
    foreach($src_vars as $i => $src_w) {
        $tgt_w = $tgt_vars[$i];
        $scaled_src = $g->scale($tau, $src_w);
        $scaled_tgt = $g->scale(1.0 - $tau, $tgt_w);
        $new_val = $g->add($scaled_src, $scaled_tgt);
        $tgt_w->assign($new_val);
    }
}

# ─────────────────────────────────────────────
# SAC + gSDE エージェント
# ─────────────────────────────────────────────
class SACGSDEAgent
{
    private Builder $nn;
    private object $la;
    private object $g;
    private int $act_dim;
    private float $act_limit;
    public GSDEActor $actor;
    public Critic $critic;
    public Critic $critic_target;
    private float $target_entropy;
    private Variable $log_alpha;
    private object $actor_opt;
    private object $critic_opt;
    private object $alpha_opt;

    public function __construct(
        Builder $nn,
        int $obs_dim,
        int $act_dim,
        float $act_limit
    )
    {
        $this->nn = $nn;
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        $this->act_dim   = $act_dim;
        $this->act_limit = $act_limit;
        $la = $this->la; 

        $this->actor         = new GSDEActor($nn, $obs_dim, $act_dim);
        $this->critic        = new Critic($nn, $obs_dim, $act_dim, HIDDEN_DIM);
        $this->critic_target = new Critic($nn, $obs_dim, $act_dim, HIDDEN_DIM);

        # ダミー入力で build してから weights をコピー
        $dummy_obs = $this->g->Variable($la->zeros($la->alloc([1, $obs_dim])));
        $dummy_act = $this->g->Variable($la->zeros($la->alloc([1, $act_dim])));
        
        $this->actor->forward_train($dummy_obs);
        $this->critic->forward($dummy_obs, $dummy_act);
        $this->critic_target->forward($dummy_obs, $dummy_act);
        
        $critic_vars = $this->critic->trainableVariables();
        $critic_vars = $this->critic->variables();


        soft_update($this->g, $this->critic, $this->critic_target, 1.0);   # 完全コピー

        $this->actor_opt  = $nn->optimizers->Adam(lr: LR_ACTOR);
        $this->critic_opt = $nn->optimizers->Adam(lr: LR_CRITIC);
        $this->alpha_opt  = $nn->optimizers->Adam(lr: LR_ALPHA);

        # 自動エントロピー調整
        # PyTorch: torch.tensor(log(ALPHA_INIT), requires_grad=True)
        # TF:      tf.Variable(..., trainable=True)
        $this->target_entropy = -(float)$act_dim;
        $this->log_alpha = $this->g->Variable(
            $this->la->array([log(ALPHA_INIT)]),
            trainable:true, name:"log_alpha"
        );
    }

    # @property
    public function alpha() : Variable
    {
        return $this->g->exp($this->log_alpha);
    }


    # ── 行動選択 ────────────────────────────────
    public function sample_noise() : Variable
    {
        return $this->actor->sample_noise();
    }


    public function select_action(NDArray $obs, Variable $W_noise) : NDArray
    {
        $obs_t  = $this->g->Variable($this->la->expandDims($obs, 0));
        $action_var = $this->actor->forward_inference($obs_t, $W_noise);
        $action = $action_var->value();
        
        $action_flat = $action->reshape([$this->act_dim]);
        $action_sc = $this->la->scal($this->act_limit, $action_flat);
        
        return $this->clip_ndarray($action_sc, -$this->act_limit, $this->act_limit);
    }
    
    private function clip_ndarray(NDArray $x, float $min, float $max) : NDArray
    {
        $arr = $x->toArray();
        array_walk_recursive($arr, function(&$v) use ($min, $max) {
            $v = max(min($v, $max), $min);
        });
        return $this->la->array($arr);
    }

    
    # ── 学習 ────────────────────────────────────
    #   各ブロックが独立した GradientTape を持つ。
    #
    #   PyTorch → TF 対応:
    #       optimizer.zero_grad()           (不要: TF は毎回新しい tape)
    #       loss.backward()              →  grads = tape.gradient(loss, vars)
    #       optimizer.step()             →  opt.apply_gradients(zip(grads, vars))
    #       with torch.no_grad():        →  tape 外 + tf.stop_gradient()
    #
    #   Rindow-NN への移植時も同じ構造で書ける。
    public function update(ReplayBuffer $buffer) : array
    {
        $g = $this->g;
        [$obs, $actions, $rewards, $next_obs, $dones] = $buffer->sample(BATCH_SIZE);

        $obs_v      = $g->Variable($obs);
        $actions_v  = $g->Variable($actions);
        $rewards_v  = $g->Variable($rewards);
        $next_obs_v = $g->Variable($next_obs);
        $dones_v    = $g->Variable($dones);

        # ── [A] target_q (勾配不要) ──────────────
        # tape 外で計算 → 自動的に勾配追跡なし
        # tf.stop_gradient で念のため勾配を遮断
        [$next_actions, $next_log_pi] = $this->actor->forward_train($next_obs_v);
        $next_actions_sc = $g->mul($next_actions, $this->act_limit);
        
        [$q1_next, $q2_next] = $this->critic_target->forward($next_obs_v, $next_actions_sc);
        $q_next_min = $g->minimum($q1_next, $q2_next);
        
        $alpha_next_log_pi = $g->mul($this->alpha(), $next_log_pi);
        $q_next = $g->sub($q_next_min, $alpha_next_log_pi);
        
        $one_minus_dones = $g->sub(1.0, $dones_v);
        $gamma_dones_q_next = $g->mul(GAMMA, $g->mul($one_minus_dones, $q_next));
        $target_q = $g->stopGradient($g->add($rewards_v, $gamma_dones_q_next));

        # ── [B] Critic 更新 ──────────────────────
        # PyTorch: critic_loss.backward(); critic_opt.step()
        $critic = $this->critic;
        $critic_loss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $critic, $obs_v, $actions_v, $target_q)
        {
            [$q1, $q2] = $critic->forward($obs_v, $actions_v);
            $critic_loss = $g->add(
                $g->reduceMean($g->square($g->sub($q1, $target_q))),
                $g->reduceMean($g->square($g->sub($q2, $target_q)))
            );
            return $critic_loss;
        });

        $critic_vars = $critic->trainableVariables();
        $critic_grads = $tape->gradient($critic_loss, $critic_vars);
        $this->critic_opt->update($critic_vars, $critic_grads);

        # ── [C] Actor 更新 ───────────────────────
        $act_limit = $this->act_limit;
        $actor = $this->actor;
        $critic = $this->critic;
        $agent = $this;
        [$actor_loss,$log_pi] = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $agent, $actor, $obs_v, $act_limit, $critic)
        {
            [$new_actions, $log_pi] = $actor->forward_train($obs_v);
            $new_actions_sc = $g->mul($new_actions, $act_limit);
            [$q1_pi, $q2_pi] = $critic->forward($obs_v, $new_actions_sc);
            $actor_loss = $g->reduceMean($g->sub($g->mul($g->stopGradient($agent->alpha()), $log_pi), $g->minimum($q1_pi, $q2_pi)));
            return [$actor_loss,$log_pi];
        });
        
        $actor_vars = $this->actor->trainableVariables();
        $actor_grads = $tape->gradient($actor_loss, $actor_vars);
        $this->actor_opt->update($actor_vars, $actor_grads);

        # ── [D] Alpha 更新 ───────────────────────
        $log_alpha = $this->log_alpha;
        $target_entropy = $this->target_entropy;
        $alpha_loss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $log_alpha, $log_pi, $target_entropy)
        {
            $alpha_loss = $g->scale(-1.0, $g->reduceMean($g->mul($log_alpha, $g->stopGradient($g->add($log_pi, $target_entropy)))));
            return $alpha_loss;
        });
        $alpha_vars = [$this->log_alpha];
        $alpha_grads = $tape->gradient($alpha_loss, $alpha_vars);
        $this->alpha_opt->update($alpha_vars, $alpha_grads);

        # ── [E] Critic ソフトアップデート ────────
        soft_update($this->g, $this->critic, $this->critic_target, TAU);

        return [
            "critic_loss" => $critic_loss->value()->toArray(),
            "actor_loss"  => $actor_loss->value()->toArray(),
            "alpha"       => $this->alpha()->value()->toArray(),
        ];
    }
}

# ─────────────────────────────────────────────
# 評価ループ
# ─────────────────────────────────────────────
function evaluate(
    Builder $nn,
    SACGSDEAgent $agent,
    int $n_episodes = EVAL_EPISODES
) : float
{
    $la = $nn->backend()->primaryLA();
    $env = new Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0($la);
    $total = 0.0;
    for ($i = 0; $i < $n_episodes; $i++) {
        [$obs, $info] = $env->reset();
        $W_noise = $agent->sample_noise();
        $done = false;
        $step = 0;
        while (!$done) {
            if ($step % GSDE_RESET_FREQ == 0) {
                $W_noise = $agent->sample_noise();
            }
            $action = $agent->select_action($obs, $W_noise);
            [$next_obs, $reward, $terminated, $truncated, $info] = $env->step($action);
            $done = $terminated || $truncated;
            $obs = $next_obs;
            $total += $reward;
            $step  += 1;
        }
    }
    return $total / $n_episodes;
}


# ─────────────────────────────────────────────
# メインループ
# ─────────────────────────────────────────────
function main()
{
    $mo = new MatrixOperator();
    $la = $mo->laRawMode();
    $nn = new NeuralNetworks($mo);

    $env = new Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0($la);
    
    $stateShape = $env->observationSpace()->shape();
    $obs_dim = $stateShape[0];
    
    $actionSpace = $env->actionSpace();
    $act_dim = $actionSpace->shape()[0];
    
    $act_limit = 1.0; 

    echo "Env: MountainCarContinuous-v0  obs_dim={$obs_dim}  act_dim={$act_dim}  act_limit={$act_limit}\n";
    echo "gSDE latent_dim=" . GSDE_LATENT_DIM . "  reset_freq=" . GSDE_RESET_FREQ . "\n";

    $agent  = new SACGSDEAgent($nn, $obs_dim, $act_dim, $act_limit);
    $buffer = new ReplayBuffer($nn, BUFFER_SIZE, $obs_dim, $act_dim);

    [$obs,$info] = $env->reset();
    $W_noise = $agent->sample_noise();

    $episode_reward = 0.0;
    $episode_step   = 0;
    $episode_count  = 0;
    $best_eval      = -INF;
    $eval_history   = [];
    $consecutive_high_scores = 0;

    for ($step = 1; $step <= TOTAL_STEPS; $step++) {

        if ($episode_step % GSDE_RESET_FREQ == 0) {
            $W_noise = $agent->sample_noise();
        }

        if ($step < START_STEPS) {
            $action = $actionSpace->sample();
        } else {
            $action = $agent->select_action($obs, $W_noise);
        }

        [$next_obs, $reward, $terminated, $truncated, $info] = $env->step($action);
        $done = $terminated || $truncated;
        $episode_reward += $reward;
        $episode_step   += 1;

        $buffer->add($obs, $action, $reward, $next_obs, $terminated);
        $obs = $next_obs;

        if ($done) {
            $episode_count += 1;
            [$obs,$info] = $env->reset();
            $W_noise = $agent->sample_noise();
            $episode_reward = 0.0;
            $episode_step   = 0;
        }

        if ($step >= START_STEPS && $step % UPDATE_EVERY == 0) {
            $agent->update($buffer);
        }

        if ($step % EVAL_EVERY == 0) {
            $mean_reward = evaluate($nn, $agent);
            $eval_history[] = $mean_reward;
            if (count($eval_history) > 10) {
                array_shift($eval_history);
            }
            $marker = ($mean_reward > $best_eval) ? " ← best" : "";
            $best_eval = max($best_eval, $mean_reward);
            
            if ($mean_reward >= 80.0) {
                $consecutive_high_scores++;
            } else {
                $consecutive_high_scores = 0;
            }
            
            $history_str = implode(", ", array_map(fn($v) => sprintf("%+.2f", $v), $eval_history));
            
            printf(
                "Step %7d | EvalReward=%+8.2f | Best=%+8.2f | Alpha=%0.4f | Consecutive80+=%d | History=[%s]%s\n",
                $step,
                $mean_reward,
                $best_eval,
                $agent->alpha()->value()->toArray()[0],
                $consecutive_high_scores,
                $history_str,
                $marker
            );
            
            if ($mean_reward >= 90.0) {
                echo "🎉 Solved! (Single evaluation mean reward >= 90)\n";
                break;
            }
            if ($consecutive_high_scores >= 3) {
                echo "🎉 Solved! (Consecutive 3 evaluations mean reward >= 80)\n";
                break;
            }
        }
    }

    echo "\nTraining finished. Best eval reward: {$best_eval}\n";

    echo "\n─────────────────────────────────────────────\n";
    echo "Testing trained model (5 episodes)\n";
    echo "─────────────────────────────────────────────\n";
    $test_episodes = 5;
    $test_rewards = [];
    for ($i = 1; $i <= $test_episodes; $i++) {
        [$obs, $info] = $env->reset();
        $W_noise = $agent->sample_noise();
        $done = false;
        $step = 0;
        $ep_reward = 0.0;
        while (!$done) {
            if ($step % GSDE_RESET_FREQ == 0) {
                $W_noise = $agent->sample_noise();
            }
            $action = $agent->select_action($obs, $W_noise);
            [$next_obs, $reward, $terminated, $truncated, $info] = $env->step($action);
            $done = $terminated || $truncated;
            $obs = $next_obs;
            $ep_reward += $reward;
            $step += 1;
        }
        $test_rewards[] = $ep_reward;
        printf("Test Episode %d: Reward = %+8.2f (Steps: %d)\n", $i, $ep_reward, $step);
    }
    $avg_test_reward = array_sum($test_rewards) / count($test_rewards);
    printf("Average Test Reward: %+8.2f\n", $avg_test_reward);
}

main();
    