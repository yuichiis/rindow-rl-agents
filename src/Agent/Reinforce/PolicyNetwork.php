<?php
namespace Rindow\RL\Agents\Agent\Reinforce;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/** Multi-layer categorical policy used by REINFORCE. */
class PolicyNetwork extends AbstractModel
{
    protected AbstractModel $policy;

    public function __construct(
        Builder $nn,
        int $obsDim,
        int $numActions,
        array $hiddenLayers = [128],
        string $activation = 'relu',
    ) {
        parent::__construct($nn);
        if ($hiddenLayers === []) {
            throw new \InvalidArgumentException('hiddenLayers must contain at least one layer.');
        }
        $layers = [];
        foreach ($hiddenLayers as $i => $units) {
            if ($units < 1) {
                throw new \InvalidArgumentException('Hidden layer sizes must be positive.');
            }
            $options = ['activation'=>$activation];
            if ($i === 0) $options['input_shape'] = [$obsDim];
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $layers[] = $nn->layers->Dense($numActions);
        $this->policy = $nn->models->Sequential($layers);
    }

    public function call(Variable $observations, ?bool $training = null) : Variable
    {
        return $this->policy->forward($observations, $training);
    }

}
