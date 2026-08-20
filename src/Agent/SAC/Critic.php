<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/**
 * ─────────────────────────────────────────────
 * Critic (Double Q)
 * ─────────────────────────────────────────────
 * Twin independent Q networks used to reduce positive value bias.
 * 
 */
class Critic extends AbstractModel
{
    public QNetwork $q1;
    public QNetwork $q2;

    /**
     * @param int|array<int,int> $obsDim
     * @param array<int,object>|null $featureLayers
     */
    public function __construct(
        Builder $nn,
        int|array $obsDim,
        int $actDim,
        int $hiddenDim,
        ?array $featureLayers=null,
    )
    {
        parent::__construct($nn);
        $this->q1 = new QNetwork($nn,$obsDim,$actDim,$hiddenDim,$featureLayers);
        $this->q2 = new QNetwork($nn,$obsDim,$actDim,$hiddenDim,$featureLayers);
    }

    /** @return array{Variable,Variable} */
    public function call(Variable $obs, Variable $action, ?bool $training=null) : array
    {
        return [$this->q1->forward($obs, $action, $training), $this->q2->forward($obs, $action, $training)];
    }
}

