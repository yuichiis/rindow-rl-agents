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
 * Correspondence with the PyTorch formulation:
 *        nn.Sequential(Linear, ReLU, ...)  → tf.keras.Sequential([Dense(...), ...])
 *        nn.Parameter(tensor)              → tf.Variable(..., trainable=True)
 * 
 */
class GSDEActor extends AbstractModel
{
    private ?Variable $lastSigmaZ = null;
    private object $la;
    private object $g;
    private int $latentsDim;
    protected Model $phiNet;  // must be protected or public to be found by trainable variables
    protected object $muHead;    // must be protected or public
    protected Variable $logStd; // must be protected of public
    
    /**
     * @param int|array<int,int> $obsDim
     * @param array<int,object>|null $featureLayers
     */
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
        
        $this->latentsDim = $latentDim;

        // Shared feature extractor (phi_net).
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

        // Mean-action head.
        $this->muHead = $nn->layers->Dense($actDim, input_shape:[$latentDim]);

        // Trainable gSDE log standard deviation.
        $this->logStd = $this->g->Variable(
            $this->la->fill(-1.0,$this->la->alloc([$actDim, $latentDim],dtype:NDArray::float32)),
            trainable:True, name:"log_std"
        );
    }

    // Shared features make both the mean and exploration scale state-dependent.
    /** @return array{Variable,Variable} */
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
     * Samples W_noise ~ N(0, std_W^2). The runner retains this matrix and
     * resamples it at the configured gSDE interval.
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
     * Inference path using a fixed exploration matrix without gradient tracking.
     *   PyTorch: with torch.no_grad(): ...
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
        // Evaluation returns tanh(mu(s)) without gSDE exploration noise.
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
        $this->la->fill($value,$this->logStd->value());
    }

    public function diagnosticSigmaZ() : ?Variable
    {
        return $this->lastSigmaZ;
    }

    /**
     * Self-contained training path. Calling it inside a GradientTape allows
     * gradients to reach log_std.
     *
     * PyTorch reparameterization:
     *     eps   = torch.randn(B, act_dim, latent_dim)
     *     W     = eps * std_W.unsqueeze(0)
     *     noise = torch.einsum("bl,bal->ba", phi, W)
     *
     * TensorFlow-equivalent reparameterization:
     *     eps   = tf.random.normal([B, act_dim, latent_dim])
     *     W     = eps * std_W[tf.newaxis, :, :]
     *     noise = tf.einsum("bl,bal->ba", phi, W)
     */
    /** @return array{Variable,Variable} */
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
        
        $logProb = $g->add(
            $g->scale(-1.0,$logSigma),$g->constant(-0.91893853320467)
        );
        $logProb = $g->sub($logProb, $term3);

        $yTSq = $g->square($yT);
        $tanhCorrInner = $g->sub(
            $g->constant(1.0+1e-6),$yTSq
        ); // Change-of-variables correction for tanh squashing.
        $tanhCorr = $g->log($tanhCorrInner);
        $logProb = $g->sub($logProb, $tanhCorr);
        
        $logProb = $g->reduceSum($logProb, axis: -1, keepdims: true);

        return [$yT, $logProb];
    }

    /**
     * The generic model call uses the differentiable training path.
     */
    /** @return array{Variable,Variable} */
    public function call(Variable $obs) : array
    {
        return $this->forwardTrain($obs);
    }

}

