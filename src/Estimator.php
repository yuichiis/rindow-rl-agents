<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Estimator
{
    public function stateShape() : array;

    public function numActions() : int;

    /**
     * Action values for each states, values mean Q-values or action-quantity or something.
     * Values ​​Including NaN as rules for selection.
     * 
     * @param  NDArray $states : (batches,...StateDims) typeof int32 or float32
     * @return NDArray $actionValues : (batches,...ValueDims) typeof float32
     */
    public function getActionValues(NDArray $states, ?bool $std=null) : NDArray|array;

    ///**
    //* @param NDArray $states   : N x StatesDims typeof int32 or float32  ( ex. [[0.0],[1.0],[2.0]] )
    //* @return NDArray $actions : N x ActionsDims typeof int32 ( ex. [[0],[1],[2]] )
    //*/
    //public function sample(NDArray $states) : NDArray;

    /**
     * Action Probabilities for random choice of actions.
     * Probabilities that do not include NaN.
     * 
     * @param  NDArray $states  : (batches,1) typeof int32 or float32
     * @return NDArray $probabilities : (batches,numActions) typeof float32
     */
    //public function probabilities(NDArray $states) : ?NDArray;
}
