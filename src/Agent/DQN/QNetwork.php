<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/** State-action value network for discrete action spaces. */
class QNetwork extends AbstractModel
{
    protected AbstractModel $model;

    public function __construct(
        Builder $nn,
        int $obsDim,
        int $numActions,
        array $hiddenLayers=[128, 128],
    ) {
        parent::__construct($nn);
        if ($hiddenLayers === []) {
            throw new \InvalidArgumentException('hiddenLayers must contain at least one layer.');
        }
        $layers = [];
        foreach ($hiddenLayers as $i=>$units) {
            if ($units < 1) throw new \InvalidArgumentException('Hidden layer sizes must be positive.');
            $options = ['activation'=>'relu'];
            if ($i === 0) $options['input_shape'] = [$obsDim];
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $layers[] = $nn->layers->Dense($numActions);
        $this->model = $nn->models->Sequential($layers);
    }

    public function call(Variable $observations, ?bool $training=null) : Variable
    {
        return $this->model->forward($observations, $training);
    }

    public function syncWeightCaches() : void
    {
        foreach ($this->model->submodules() as $module) {
            $module->reverseSyncWeightVariables();
        }
    }
}
