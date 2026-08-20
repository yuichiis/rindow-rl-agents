<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/** Deterministic policy used by DDPG. The output is normalized to [-1, 1]. */
class Actor extends AbstractModel
{
    protected AbstractModel $model;

    public function __construct(
        Builder $nn,
        int|array $obsDim,
        int $actDim,
        int $hiddenDim,
        ?array $featureLayers=null,
    ) {
        parent::__construct($nn);
        if ($featureLayers === []) $featureLayers = null;
        $inputShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        $layers = $featureLayers === null
            ? []
            : array_map(static fn(object $layer)=>clone $layer,$featureLayers);
        $firstOptions = ['activation'=>'relu'];
        if ($featureLayers === null) $firstOptions['input_shape'] = $inputShape;
        $layers[] = $nn->layers->Dense($hiddenDim,...$firstOptions);
        $layers[] = $nn->layers->Dense($hiddenDim,activation:'relu');
        $layers[] = $nn->layers->Dense($actDim,activation:'tanh');
        $this->model = $nn->models->Sequential($layers);
    }

    public function call(Variable $obs, ?bool $training=null) : Variable
    {
        return $this->model->forward($obs, $training);
    }

}
