<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/**
 * ─────────────────────────────────────────────
 * Critic (Double Q)
 * ─────────────────────────────────────────────
 *   PyTorch の Critic(q1, q2) に対応。
 *   TF では Functional API で 2 つの独立したサブモデルを保持する。
 * 
 */
class Critic extends AbstractModel
{
    public QNetwork $q1;
    public QNetwork $q2;

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

    public function call(Variable $obs, Variable $action, ?bool $training=null) : array
    {
        return [$this->q1->forward($obs, $action, $training), $this->q2->forward($obs, $action, $training)];
    }

    public function syncWeightCaches() : void
    {
        $this->q1->syncWeightCaches();
        $this->q2->syncWeightCaches();
    }
}

