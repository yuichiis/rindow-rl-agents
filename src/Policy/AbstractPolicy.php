<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Util\Random;

abstract class AbstractPolicy implements Policy
{
    use Random;

    protected object $la;

    public function __construct($la)
    {
        $this->la = $la;
    }

    public function register(?EventManager $eventManager=null) : void
    {
    }

    /**
     * Select the action with the highest value among the action values ​​received from the estimator.
     * Action Values ​​Including NaN.
     * 
     * param  NDArray $states  : (batches,...StateDims) typeof int32 or float32
     * return NDArray $actions : (batches) typeof int32
     */
    protected function calcMaxValueActions(Estimator $estimator, NDArray $states, ?NDArray $masks) : NDArray
    {
        $la = $this->la;
        $actionValues = $estimator->getActionValues($states); // (batches,numActions)

        //$actionValues = $la->nan2num($la->copy($actionValues),alpha:-INF);
        if($masks) {
            $actionValues = $la->masking($masks,$actionValues,fill:-INF);
        }
        $actions = $la->reduceArgMax($actionValues,axis:-1); // (batches)
        return $actions;
    }

    ///**
    //* param  NDArray $states  : (batches,...StateDims) typeof int32 or float32
    //* return NDArray $actions : (batches) typeof int32
    //*/
    //protected function calcRandomActions(Estimator $estimator, NDArray $states, ?NDArray $masks) : NDArray
    //{
    //    $la = $this->la;
    //    $valids = $estimator->probabilities($states);       // (batches,...ValueDims)
    //    //echo "shape[".implode(',',$logits->shape())."],";
    //    //echo "p[".implode(',',$logits->toArray()[0])."],";
    //    if($logits===null) {
    //        $numActions = $estimator->numActions();
    //        $batches = count($states);
    //        $actions = $la->randomUniform([$batches],0,$numActions-1,dtype:NDArray::int32);
    //    } else {
    //        $actions = $this->randomCategorical($logits);
    //    }
    //    return $actions;
    //}

    /**
    * param  NDArray $states  : (batches,...StateDims) typeof int32 or float32
    * return NDArray $actions : (batches) typeof int32
    */
    protected function calcRandomActions(Estimator $estimator, NDArray $states, ?NDArray $masks) : NDArray
    {
        $la = $this->la;
        $numActions = $estimator->numActions();
        $batches = count($states);
        if($masks===null) {
            $actions = $la->randomUniform([$batches],0,$numActions-1,dtype:NDArray::int32);
        } else {
            $logits = $la->masking($masks,$la->log($la->ones($la->alloc([$batches,$numActions]))),fill:-INF);
            $actions = $this->randomCategorical($logits);
        }
        return $actions;
    }
}   