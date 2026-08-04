<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/** A shared feature extractor with categorical policy and value heads. */
class ActorCritic extends AbstractModel
{
    // AbstractModel discovers protected child modules when collecting weights.
    protected AbstractModel $features;
    protected object $policyHead;
    protected object $valueHead;

    public function __construct(
        Builder $nn,
        int $obsDim,
        int $numActions,
        array $hiddenLayers = [64, 64],
    ) {
        parent::__construct($nn);
        if ($hiddenLayers === []) {
            throw new \InvalidArgumentException('hiddenLayers must contain at least one layer.');
        }
        $layers = [];
        foreach ($hiddenLayers as $i => $units) {
            $options = ['activation'=>'tanh'];
            if ($i === 0) {
                $options['input_shape'] = [$obsDim];
            }
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $this->features = $nn->models->Sequential($layers);
        $this->policyHead = $nn->layers->Dense($numActions);
        $this->valueHead = $nn->layers->Dense(1);
    }

    /** @return array{Variable,Variable} unnormalised action logits and V(s) */
    public function call(Variable $observations, ?bool $training = null) : array
    {
        $features = $this->features->forward($observations, $training);
        return [
            $this->policyHead->forward($features, $training),
            $this->valueHead->forward($features, $training),
        ];
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
