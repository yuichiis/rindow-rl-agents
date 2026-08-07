<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;

/** A shared feature extractor with categorical policy and value heads. */
class ActorCritic extends AbstractModel
{
    // AbstractModel discovers protected child modules when collecting weights.
    protected AbstractModel $features;
    protected object $policyHead;
    protected ?Variable $logStd = null;
    protected object $valueHead;
    private Builder $nnBuilder;
    private ?NDArray $actionScale = null;
    private ?NDArray $actionShift = null;

    public function __construct(
        Builder $nn,
        int|array $obsDim,
        int $numActions,
        array $hiddenLayers = [64, 64],
        private bool $continuous = false,
        float $initialLogStd = -0.5,
        ?NDArray $actionMin = null,
        ?NDArray $actionMax = null,
        mixed $actionKernelInitializer = null,
        string $activation = 'tanh',
        ?array $featureLayers = null,
    ) {
        parent::__construct($nn);
        $this->nnBuilder = $nn;
        if ($hiddenLayers === []) {
            throw new \InvalidArgumentException('hiddenLayers must contain at least one layer.');
        }
        $inputShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        $layers = $featureLayers === null
            ? []
            : array_map(static fn(object $layer)=>clone $layer,$featureLayers);
        foreach ($hiddenLayers as $i => $units) {
            $options = ['activation'=>$activation];
            if ($i === 0 && $featureLayers === null) {
                $options['input_shape'] = $inputShape;
            }
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $this->features = $nn->models->Sequential($layers);
        if ($continuous) {
            $this->policyHead = $nn->layers->Dense(
                $numActions,
                activation:'tanh',
                kernel_initializer:$actionKernelInitializer,
            );
            if ($actionMin !== null && $actionMax !== null) {
                $this->actionScale = $nn->gradient()->constant(
                    $nn->backend()->primaryLA()->scal(
                        0.5,
                        $nn->backend()->primaryLA()->axpy(
                            $actionMin, $nn->backend()->primaryLA()->copy($actionMax), -1.0
                        )
                    )
                );
                $this->actionShift = $nn->gradient()->constant(
                    $nn->backend()->primaryLA()->scal(
                        0.5,
                        $nn->backend()->primaryLA()->axpy(
                            $actionMin, $nn->backend()->primaryLA()->copy($actionMax), 1.0
                        )
                    )
                );
            }
            $this->logStd = $nn->gradient()->Variable(
                $nn->backend()->primaryLA()->fill(
                    $initialLogStd,
                    $nn->backend()->primaryLA()->alloc([$numActions], dtype:NDArray::float32)
                ),
                trainable:true,
                name:'a2c_gaussian_log_std',
            );
        } else {
            $this->policyHead = $nn->layers->Dense($numActions);
        }
        $this->valueHead = $nn->layers->Dense(1);
    }

    /** Discrete: logits/value. Continuous: mean/value/log standard deviation. */
    public function call(Variable $observations, ?bool $training = null) : array
    {
        $features = $this->features->forward($observations, $training);
        $action = $this->policyHead->forward($features, $training);
        if ($this->continuous && $this->actionScale !== null) {
            $g = $this->nnBuilder->gradient();
            $action = $g->add($g->mul($action, $this->actionScale), $this->actionShift);
        }
        $outputs = [
            $action,
            $this->valueHead->forward($features, $training),
        ];
        if ($this->continuous) {
            // Keep the global parameter one-dimensional. Gaussian operations
            // broadcast it across the batch, matching the legacy A2C exactly.
            $outputs[] = $this->logStd;
        }
        return $outputs;
    }

    /** Keep layer-side weight caches current after an optimiser update. */
    public function syncWeightCaches() : void
    {
        foreach ([$this->features, $this->policyHead, $this->valueHead] as $model) {
            foreach ($model->submodules() as $module) {
                $module->reverseSyncWeightVariables();
            }
        }
    }
}
