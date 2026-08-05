<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/** Deterministic policy used by DDPG. The output is normalized to [-1, 1]. */
class Actor extends AbstractModel
{
    protected AbstractModel $model;

    public function __construct(Builder $nn, int $obsDim, int $actDim, int $hiddenDim)
    {
        parent::__construct($nn);
        $this->model = $nn->models->Sequential([
            $nn->layers->Dense($hiddenDim, activation:'relu', input_shape:[$obsDim]),
            $nn->layers->Dense($hiddenDim, activation:'relu'),
            $nn->layers->Dense($actDim, activation:'tanh'),
        ]);
    }

    public function call(Variable $obs, ?bool $training=null) : Variable
    {
        return $this->model->forward($obs, $training);
    }

    public function syncWeightCaches() : void
    {
        foreach ($this->model->submodules() as $module) {
            $module->reverseSyncWeightVariables();
        }
    }
}
