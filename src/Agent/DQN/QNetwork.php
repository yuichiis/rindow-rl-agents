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
        int|array $obsDim,
        int $numActions,
        array $hiddenLayers=[128, 128],
        ?array $featureLayers=null,
    ) {
        parent::__construct($nn);
        if ($featureLayers === []) $featureLayers = null;
        if ($hiddenLayers === [] && $featureLayers === null) {
            throw new \InvalidArgumentException('hiddenLayers must contain at least one layer.');
        }
        $inputShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        // Each QNetwork must own independent layer instances.  The online and
        // target networks are constructed from the same feature layer template.
        $layers = $featureLayers === null
            ? []
            : array_map(static fn(object $layer)=>clone $layer,$featureLayers);
        foreach ($hiddenLayers as $i=>$units) {
            if ($units < 1) throw new \InvalidArgumentException('Hidden layer sizes must be positive.');
            $options = ['activation'=>'relu'];
            if ($i === 0 && $featureLayers === null) $options['input_shape'] = $inputShape;
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $layers[] = $nn->layers->Dense($numActions);
        $this->model = $nn->models->Sequential($layers);
    }

    public function call(Variable $observations, ?bool $training=null) : Variable
    {
        return $this->model->forward($observations, $training);
    }

}
