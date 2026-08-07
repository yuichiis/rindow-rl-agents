<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Layer\Layer;

/**
 * gSDE Actor
 * 
 *    PyTorch 版との対応:
 *        nn.Sequential(Linear, ReLU, ...)  → tf.keras.Sequential([Dense(...), ...])
 *        nn.Parameter(tensor)              → tf.Variable(..., trainable=True)
 * 
 */
class GSDEActor extends AbstractModel
{
    private ?Variable $lastSigmaZ = null;
    private object $la;
    private object $g;
    private int $actDim;
    private int $latentsDim;
    protected Model $phiNet;  // must be protected or public to be found by trainable variables
    protected Layer $muHead;    // must be protected or public
    protected Variable $logStd; // must be protected of public
    
    public function __construct(
        Builder $nn,
        int|array $obsDim,
        int $actDim,
        int $latentDim,
        int $hiddenDim,
        ?array $featureLayers=null,
    )
    {
        parent::__construct($nn);
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        
        $this->actDim    = $actDim;
        $this->latentsDim = $latentDim;

        # 共有特徴抽出器  (PyTorch: phi_net)
        if ($featureLayers === []) $featureLayers = null;
        $layers = $featureLayers === null
            ? []
            : array_map(static fn(object $layer)=>clone $layer,$featureLayers);
        $firstOptions = ['activation'=>'relu'];
        if ($featureLayers === null) {
            $firstOptions['input_shape'] = is_int($obsDim)
                ? [$obsDim] : array_values($obsDim);
        }
        $layers[] = $nn->layers->Dense($hiddenDim,...$firstOptions);
        $layers[] = $nn->layers->Dense($latentDim,activation:'relu');
        $this->phiNet = $nn->models->Sequential($layers);

        // 平均ヘッド  (PyTorch: mu_head = nn.Linear)
        $this->muHead = $nn->layers->Dense($actDim, input_shape:[$latentDim]);

        // gSDE 対数標準偏差  (PyTorch: nn.Parameter)
        $this->logStd = $this->g->Variable(
            $this->la->fill(-1.0,$this->la->alloc([$actDim, $latentDim],dtype:NDArray::float32)),
            trainable:True, name:"log_std"
        );
    }

    // ── 共通特徴抽出 ────────────────────────────
    private function phiAndMu(Variable $obs) : array
    {
        $phi = $this->phiNet->forward($obs);    # (B, latent_dim)
        $mu  = $this->muHead->forward($phi);    # (B, act_dim)
        return [$phi, $mu];
    }

    private function stdW() : Variable
    {
        return $this->g->exp($this->logStd);   # (act_dim, latent_dim)
    }

    /**
     * ── ① ノイズサンプル ────────────────────────
     *    W_noise ~ N(0, std_W²) をサンプルして返す。
     *    ループ変数として保持し、GSDE_RESET_FREQ ごとに再呼び出し。
     *
     *    PyTorch: torch.randn_like(std) * std
     *    TF:      tf.random.normal(tf.shape(std)) * std
     */
    public function sampleNoise() : Variable
    {
        $g = $this->g;
        $std = $this->stdW();
        $eps = $g->randomNormal($std);
        return $g->mul($eps, $std);  # (act_dim, latent_dim)
    }

    /**
     * ── ② 推論パス（勾配なし） ──────────────────
     *   PyTorch: with torch.no_grad(): ...
     *   TF: tape 外から呼ぶことで自動的に勾配追跡なし
     */
    public function forwardInference(Variable $obs, Variable $wNoise) : Variable
    {
        [$phi, $mu] = $this->phiAndMu($obs);
        # (act_dim, latent_dim) @ (latent_dim, 1) → (act_dim, 1) → (1, act_dim)
        $phiT = $this->g->transpose($phi);
        $matmul = $this->g->matmul($wNoise, $phiT);
        $noise = $this->g->transpose($matmul); # (1, act_dim)
        return $this->g->tanh($this->g->add($mu, $noise));
    }

    public function forwardDeterministic(Variable $obs) : Variable
    {
        // 評価用: gSDE の探索ノイズを使わず、tanh(mu(s)) を返す。
        [, $mu] = $this->phiAndMu($obs);
        return $this->g->tanh($mu);
    }

    public function diagnosticMu(Variable $obs) : Variable
    {
        [, $mu] = $this->phiAndMu($obs);
        return $mu;
    }

    public function diagnosticPhi(Variable $obs) : Variable
    {
        [$phi,] = $this->phiAndMu($obs);
        return $phi;
    }

    public function diagnosticLogStd() : Variable
    {
        return $this->logStd;
    }

    public function resetLogStd(float $value = -1.0) : void
    {
        $this->logStd->assign($this->la->fill($value, $this->la->alloc($this->logStd->shape(), dtype:NDArray::float32)));
    }

    public function syncWeightCaches() : void
    {
        foreach ($this->phiNet->submodules() as $module) {
            $module->reverseSyncWeightVariables();
        }
        $this->muHead->reverseSyncWeightVariables();
    }

    public function diagnosticSigmaZ() : ?Variable
    {
        return $this->lastSigmaZ;
    }

    /**
     * ── ③ 学習パス（GradientTape 内で呼ぶ） ─────
     * 外部状態に依存しない自己完結パス。
     * GradientTape スコープ内で呼ぶことで log_std への勾配が流れる。
     *
     * PyTorch の reparameterization:
     *     eps   = torch.randn(B, act_dim, latent_dim)
     *     W     = eps * std_W.unsqueeze(0)
     *     noise = torch.einsum("bl,bal->ba", phi, W)
     *
     * TF の reparameterization:
     *     eps   = tf.random.normal([B, act_dim, latent_dim])
     *     W     = eps * std_W[tf.newaxis, :, :]
     *     noise = tf.einsum("bl,bal->ba", phi, W)
     */
    public function forwardTrain(Variable $obs) : array
    {
        $g = $this->g;
        [$phi, $mu] = $this->phiAndMu($obs);
        $stdW   = $this->stdW();                      # (act_dim, latent_dim)

        $B     = $obs->shape()[0];
        $eps   = $g->randomNormal($stdW,batchShape:[$B]);
        $W     = $g->mul($eps, $stdW);  # (B, act_dim, latent_dim) eps <- broadcast $std_W
        
        // noise = einsum("bl,bal->ba", phi, W)
        $phiReshaped = $g->reshape($phi, [$B, $this->latentsDim, 1]);
        $matmul = $g->matmul($W, $phiReshaped);
        $noise = $g->squeeze($matmul, 2);         # (B, act_dim)

        $xT = $g->add($mu, $noise);
        $yT = $g->tanh($xT);

        // sigma_z(s) = sqrt( std_W² @ phi² )
        // PyTorch: (std_W.pow(2) @ phi.T.pow(2)).sqrt().T
        // TF:      tf.transpose( tf.sqrt(std_W**2 @ tf.transpose(phi**2)) )
        $stdWSq = $g->square($stdW);
        $phiSq = $g->square($phi);
        $phiSqT = $g->transpose($phiSq);
        $matmulSq = $g->matmul($stdWSq, $phiSqT);
        $sqrt = $g->sqrt($matmulSq);
        $sigmaZ = $g->transpose($sqrt);
        $sigmaZ = $g->maximum($sigmaZ,$g->constant(1e-6));
        $this->lastSigmaZ = $sigmaZ;

        $logSigma = $g->log($sigmaZ);
        $diff = $g->sub($xT, $mu);
        $diffSq = $g->square($diff);
        $sigmaZSq = $g->square($sigmaZ);
        $twoSigmaZSq = $g->mul(2.0, $sigmaZSq);
        $term3 = $g->div($diffSq, $twoSigmaZSq);
        
        $logProb = $g->sub(-0.91893853320467, $logSigma);
        $logProb = $g->sub($logProb, $term3);

        $yTSq = $g->square($yT);
        $tanhCorrInner = $g->add($g->sub(1.0, $yTSq), 1e-6); # tanh 補正
        $tanhCorr = $g->log($tanhCorrInner);
        $logProb = $g->sub($logProb, $tanhCorr);
        
        $logProb = $g->reduceSum($logProb, axis: -1, keepdims: true);

        return [$yT, $logProb];
    }

    /**
     * tf.keras.Model の call は forward_train を使う
     */
    public function call(Variable $obs) : array
    {
        return $this->forwardTrain($obs);
    }

}

