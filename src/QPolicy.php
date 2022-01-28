<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface QPolicy
{
    public function obsSize();

    public function numActions() : int;
    /**
    * @param NDArray $state
    * @return NDArray $qValues
    */
    public function getQValues($state) : NDArray;
    public function sample($state);
}
