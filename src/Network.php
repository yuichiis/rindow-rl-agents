<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Network
{
    public function obsSize();

    public function numActions() : int;

    public function copyWeights($source,float $tau=null) : void;

    public function getQValuesBatch(NDArray $observations) : NDArray;
    //public function model();
}
