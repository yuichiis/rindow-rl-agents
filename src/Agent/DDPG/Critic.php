<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/** State-action value network. */
class Critic extends AbstractModel
{
    private object $g;
    protected AbstractModel $model;

    public function __construct(Builder $nn, int $obsDim, int $actDim, int $hiddenDim)
    {
        parent::__construct($nn);
        $this->g = $nn->gradient();
        $this->model = $nn->models->Sequential([
            $nn->layers->Dense($hiddenDim, activation:'relu', input_shape:[$obsDim+$actDim]),
            $nn->layers->Dense($hiddenDim, activation:'relu'),
            $nn->layers->Dense(1),
        ]);
    }

    public function call(Variable $obs, Variable $action, ?bool $training=null) : Variable
    {
        return $this->model->forward($this->g->concat([$obs, $action], axis:-1), $training);
    }

    public function syncWeightCaches() : void
    {
        foreach ($this->model->submodules() as $module) {
            $module->reverseSyncWeightVariables();
        }
    }
}
