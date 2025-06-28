<?php
namespace Rindow\RL\Agents\Estimator;

use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Util\Random;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;

abstract class AbstractEstimatorNetwork extends AbstractNetwork implements Estimator
{
    use Random;

    protected object $la;
    //protected ?NDArray $probabilities=null;
    //protected ?NDArray $masks;

    /**
     * $rules(numStates,numAction) : 1.0 or NaN
     * $prob(numStates,numAction)  : 0.0 <= x <= 1.0
     */
    //public function generateProbabilities(NDArray $rules) : NDArray
    //{
    //    $la = $this->la;
    //    if($rules->ndim()!=2) {
    //        throw new InvalidArgumentException('rules must be 2D NDArray');
    //    }
    //    // p[i] = exp(rules[i])/sum(exp(rules[i]))
    //    $prob = $la->exp($la->copy($rules));
    //    $prob = $la->nan2num($prob);
    //    $sum = $la->reduceSum($prob,axis:1);
    //    $la->multiply($la->reciprocal($sum),$prob,trans:true);
    //    return $prob;
    //}

    ///**
    // * @param array<int> $numActions
    // */
    //protected function initializeRules(?NDArray $rules, int $numActions) : void
    //{
    //    $la = $this->la;
    //    if($rules) {
    //        if($rules->shape()[1]!=$numActions) {
    //            throw new InvalidArgumentException('The rules must match numActions');
    //        }
    //        //$p = $this->generateProbabilities($rules);
    //        //$this->probabilities = $p;
    //        //$this->thresholds = $this->generateThresholds($p);
    //        $this->masks = $rules;
    //        //$this->masks = $la->nan2num($la->copy($rules),-INF);
    //    } else {
    //    //    $p = $la->alloc([1,$numActions]);
    //    //    $this->onesProb = $la->ones($p);
    //        //$this->probabilities = null;
    //        $this->masks = null;
    //    }
    //}

    /**
    * $states : (batches,...stateShape) typeof int32 or float32
    * $actionValues : (batches,...numActions)  typeof float32
    */
    public function getActionValues(NDArray $states) : NDArray
    {
        $la = $this->la;
        if($states->ndim()<2) {
            $specs = $la->dtypeToString($states->dtype())."(".implode(',',$states->shape()).")";
            throw new InvalidArgumentException("states must be a 2-dimensional array or higher. $specs given.");
        }
        $orgStates = $states;
        if($la->isInt($states)) {
            $states = $la->astype($states,NDArray::float32);
        }
        $values = $this->model->forward($states,false);
        //if($this->masks) {
        //    $states = $orgStates;
        //    if($states->ndim()!==2||!$la->isInt($states)||$states->shape()[1]!=1) {
        //        $specs = $la->dtypeToString($states->dtype())."(".implode(',',$states->shape()).")";
        //        throw new InvalidArgumentException("When using the rules feature, states must have one channel. $specs given.");
        //    }
        //    $states = $la->squeeze($states,axis:-1);
        //    if($states->dtype()!==NDArray::int32) {
        //        $states = $la->astype($states,NDArray::int32);
        //    }
        //    //$masks = $la->gather($this->masks,$states,$axis=null);
        //    $masks = $la->gatherb($this->masks,$states);
        //    //$la->multiply($masks,$values);
        //    $la->masking($mask,$values,fill:-INF);
        //    //$la->nan2num($values,-INF);
        //    //$la->add($masks,$values);   // 
        //}
        return $values;
    }

    /**
     * $states  : (batches,1)            : float32|int32
     * $probabilities : (batches,numActions)   : float32
     */
    //public function probabilities(NDArray $states) : ?NDArray
    //{
    //    $la = $this->la;
    //    if($this->probabilities===null) {
    //        return null;
    //    }
    //    if($states->ndim()!==2||!$la->isInt($states)||$states->shape()[1]!=1) {
    //        $specs = $la->dtypeToString($states->dtype())."(".implode(',',$states->shape()).")";
    //        throw new InvalidArgumentException("states must have one channel. $specs given.");
    //    }
    //    if($states!==NDArray::int32) {
    //        $states = $la->astype($states,NDArray::int32);
    //    }
    //    $states = $la->squeeze($states,axis:-1);
    //    $probabilities = $la->gatherb($this->probabilities,$states);
    //    return $probabilities;
    //}

}