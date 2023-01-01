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
    * @param NDArray $states   : N x StatesDims typeof int32 or float32 ( ex. [[0.0],[1.0],[2.0]] )
    * @return NDArray $qValues : N x ValuesDims typeof float32 ( ex. [[0.0,1.0],[0.5,1.5],[1.0,2.0]] )
    */
    public function getQValues(NDArray $states) : NDArray;

    /**
    * @param NDArray $states   : N x StatesDims typeof int32 or float32  ( ex. [[0.0],[1.0],[2.0]] )
    * @return NDArray $actions : N x ActionsDims typeof int32 ( ex. [[0],[1],[2]] )
    */
    public function sample(NDArray $states) : NDArray;
}
