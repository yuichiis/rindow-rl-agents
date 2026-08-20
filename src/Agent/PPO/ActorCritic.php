<?php
namespace Rindow\RL\Agents\Agent\PPO;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;

/** Actor-critic network shared by categorical, Gaussian, and gSDE policies. */
class ActorCritic extends AbstractModel
{
    private Builder $nnBuilder;
    protected ?AbstractModel $trunk = null;
    protected ?AbstractModel $actor = null;
    protected ?AbstractModel $critic = null;
    protected ?object $actorHead = null;
    protected ?object $logStdHead = null;
    protected ?Variable $sdeLogStd = null;
    protected ?object $criticHead = null;

    /**
     * @param int|array<int,int> $obsDim
     * @param array<int,int> $hiddenLayers
     * @param array<int,object>|null $featureLayers
     */
    public function __construct(
        Builder $nn,
        int|array $obsDim,
        int $numActions,
        array $hiddenLayers = [64, 64],
        private bool $sharedBackbone = false,
        private bool $continuous = false,
        private bool $useSDE = false,
        float $sdeInitialLogStd = -2.0,
        ?array $featureLayers = null,
    ) {
        parent::__construct($nn);
        $this->nnBuilder = $nn;
        if (!$sharedBackbone) {
            $this->actor = $this->mlp($nn, $obsDim, $hiddenLayers, $numActions, 'tanh');
            $this->critic = $this->mlp($nn, $obsDim, $hiddenLayers, 1, 'tanh');
            return;
        }
        $inputShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        $layers = $featureLayers === null
            ? []
            : array_map(static fn(object $layer)=>clone $layer,$featureLayers);
        foreach ($hiddenLayers as $i => $units) {
            // The legacy ActorCriticNetwork's MLP uses he_uniform for its
            // ReLU hidden layers.  Keep this explicit; Dense's library
            // default is different and changes the seeded initial policy.
            $options = ['activation' => 'relu', 'kernel_initializer' => 'he_uniform'];
            if ($i === 0 && $featureLayers === null) {
                $options['input_shape'] = $inputShape;
            }
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $this->trunk = $nn->models->Sequential($layers);
        $this->actorHead = $nn->layers->Dense($numActions);
        if ($continuous && !$useSDE) {
            $this->logStdHead = null;
        }
        if ($useSDE) {
            $latentDim = $hiddenLayers[count($hiddenLayers)-1];
            $this->sdeLogStd = $nn->gradient()->Variable(
                $nn->backend()->primaryLA()->fill(
                    $sdeInitialLogStd,
                    $nn->backend()->primaryLA()->alloc([$numActions, $latentDim], dtype:NDArray::float32)
                ),
                trainable:true,
                name:'ppo_sde_log_std',
            );
        }
        $this->criticHead = $nn->layers->Dense(1);
    }

    /**
     * @param int|array<int,int> $obsDim
     * @param array<int,int> $hiddenLayers
     */
    private function mlp(
        Builder $nn,
        int|array $obsDim,
        array $hiddenLayers,
        int $outputDim,
        string $activation,
    ) : AbstractModel {
        $layers = [];
        foreach ($hiddenLayers as $i => $units) {
            $options = ['activation' => $activation];
            if ($i === 0) {
                $options['input_shape'] = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
            }
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $layers[] = $nn->layers->Dense($outputDim);
        return $nn->models->Sequential($layers);
    }

    private function features(Variable $observations, ?bool $training = null) : Variable
    {
        return $this->trunk->forward($observations, $training);
    }

    public function policy(Variable $observations, ?bool $training = null) : Variable
    {
        if (!$this->sharedBackbone) {
            return $this->actor->forward($observations, $training);
        }
        return $this->actorHead->forward($this->features($observations, $training), $training);
    }

    public function value(Variable $observations, ?bool $training = null) : Variable
    {
        if (!$this->sharedBackbone) {
            return $this->critic->forward($observations, $training);
        }
        return $this->criticHead->forward($this->features($observations, $training), $training);
    }

    /** @return array<int,Variable> */
    public function call(Variable $observations, ?bool $training = null) : array
    {
        if (!$this->sharedBackbone) {
            return [
                $this->actor->forward($observations, $training),
                $this->critic->forward($observations, $training),
            ];
        }
        $features = $this->features($observations, $training);
        $mean = $this->actorHead->forward($features, $training);
        $value = $this->criticHead->forward($features, $training);
        if ($this->continuous) {
            if ($this->useSDE) {
                $g = $this->nnBuilder->gradient();
                $logStd = $g->clipByValue($this->sdeLogStd, -5.0, 2.0);
                $variance = $g->matmul($g->square($features), $g->transpose($g->square($g->exp($logStd))));
                $std = $g->sqrt($g->maximum(
                    $variance,$g->constant(1.0e-8)
                ));
                return [$mean, $value, $std];
            }
            if ($this->logStdHead === null) {
                $this->logStdHead = $this->nnBuilder->layers->Dense(
                    $mean->value()->shape()[1], kernel_initializer:'zeros'
                );
            }
            $logStd = $this->logStdHead->forward($features, $training);
            return [$mean, $value, $logStd];
        }
        return [$mean, $value];
    }

    public function sampleSDENoise() : NDArray
    {
        if (!$this->useSDE || $this->sdeLogStd === null) {
            throw new \LogicException('gSDE is not enabled.');
        }
        $la = $this->nnBuilder->backend()->primaryLA();
        $logStd = $la->minimum(
            $la->maximum($la->copy($this->sdeLogStd->value()), -5.0),
            2.0
        );
        return $la->multiply(
            $la->randomNormal($logStd->shape(), 0.0, 1.0),
            $la->exp($la->copy($logStd)),
        );
    }

    /** @return array{Variable,Variable,Variable,Variable} action, value, marginal std, mean */
    public function forwardSDE(Variable $observations, NDArray $noise, ?bool $training = false) : array
    {
        if (!$this->useSDE) {
            throw new \LogicException('gSDE is not enabled.');
        }
        $g = $this->nnBuilder->gradient();
        $features = $this->features($observations, $training);
        $mean = $this->actorHead->forward($features, $training);
        $value = $this->criticHead->forward($features, $training);
        $noiseTerm = $g->matmul($features, $g->transpose($g->constant($noise)));
        $logStd = $g->clipByValue($this->sdeLogStd, -5.0, 2.0);
        $variance = $g->matmul($g->square($features), $g->transpose($g->square($g->exp($logStd))));
        $std = $g->sqrt($g->maximum(
            $variance,$g->constant(1.0e-8)
        ));
        return [$g->add($mean, $noiseTerm), $value, $std, $mean];
    }

}
