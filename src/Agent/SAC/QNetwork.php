<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

class QNetwork extends AbstractModel
{
    private object $g;
    protected ?AbstractModel $featureModel = null;
    protected AbstractModel $model; // must be protected or public to be found by trainable variables

    public function __construct(
        Builder $nn,
        int|array $obsDim,
        int $actDim,
        int $hiddenDim,
        ?array $featureLayers=null,
    )
    {
        parent::__construct($nn);
        $this->g = $nn->gradient();
        if ($featureLayers === []) $featureLayers = null;
        if ($featureLayers !== null) {
            $this->featureModel = $nn->models->Sequential(
                array_map(static fn(object $layer)=>clone $layer,$featureLayers)
            );
        }
        $firstOptions = ['activation'=>'relu'];
        if ($featureLayers === null) {
            $firstOptions['input_shape'] = [
                array_product(is_int($obsDim) ? [$obsDim] : $obsDim)+$actDim
            ];
        }
        $this->model = $nn->models->Sequential([
            $nn->layers->Dense($hiddenDim,...$firstOptions),
            $nn->layers->Dense($hiddenDim, activation: 'relu'),
            $nn->layers->Dense(1),
        ]);
    }

    public function call(Variable $obs, Variable $action, ?bool $training=null) : Variable
    {
        if ($this->featureModel !== null) {
            $obs = $this->featureModel->forward($obs,$training);
        }
        $x = $this->g->concat([$obs, $action], axis: -1);
        return $this->model->forward($x, $training);
    }

}
