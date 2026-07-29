<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

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

    public function sync_weight_caches() : void
    {
        foreach ($this->model->submodules() as $module) {
            $module->reverseSyncWeightVariables();
        }
    }
}
