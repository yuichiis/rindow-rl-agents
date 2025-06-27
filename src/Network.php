<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Network
{
    public function stateShape() : array;

    public function actionShape() : array;

    public function numActions() : int;

    public function copyWeights(Network $source, ?float $tau=null) : void;

    public function builder() : object;
}
