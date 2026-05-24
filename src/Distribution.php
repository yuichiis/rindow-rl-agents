<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;

interface Distribution
{
    public function logProb(Variable $value) : Variable;

    public function entropy() : Variable;

    /**
     * @param array<int> $batchShape
     */
    public function sample(?array $batchShape=null) : Variable;
}
