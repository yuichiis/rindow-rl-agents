<?php
namespace Rindow\RL\Agents\Estimator;

use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Table;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

abstract class AbstractTable implements Estimator,Table
{
    use Random;

    const MODEL_FILENAME = '%s.model';

    protected object $la;
    //protected ?NDArray $masks;
    protected int $numStates;
    protected int $numActions;
    //protected ?NDArray $probabilities;
    protected NDArray $table;

    abstract public function initialize() : void;

    /**
     * $rules: (numStates,numActions)
     */
    public function __construct(
        object $la,
        //?NDArray $rules=null,
        int $numStates,
        int $numActions,
        )
    {
        $this->la = $la;
        //$this->masks = $rules;
        //if($rules instanceof NDArray) {
        //    if($rules->ndim()!=2) {
        //        throw new InvalidArgumentException('rules must be 2D integer');
        //    }
        //    [$numStates,$numActions] = $rules->shape();
        //    $this->numStates = $numStates;
        //    $this->numActions = $numActions;
        //} else {
            //if(!is_int($numStates) || !is_int($numActions)) {
            //    throw new InvalidArgumentException('numStates and numActions must be specified.');
            //}
            $this->numStates = $numStates;
            $this->numActions = $numActions;
        //}
    }

    public function stateShape() : array
    {
        return [1];
    }

    public function numActions() : int
    {
        return $this->numActions;
    }

    public function table() : NDArray
    {
        return $this->table;
    }

    //public function getActionValues($state) : NDArray
    //{
    //    if(!is_int($state)) {
    //        throw new InvalidArgumentException('state must be integer');
    //    }
    //    return $this->table[$state];
    //}

    /**
     * Action Values ​​Including -INF
     * 
     * $states : (batches,1)          typeof int32
     * $actionValues : (batches,numActions) typeof float32
     */
    public function getActionValues(NDArray $states) : NDArray
    {
        $la = $this->la;
        if($states->ndim()!==2||!$la->isInt($states)||$states->shape()[1]!=1) {
            $specs = $la->dtypeToString($states->dtype())."(".implode(',',$states->shape()).")";
            throw new InvalidArgumentException("states must have one channel. $specs given.");
        }
        $orgStates = $states;
        $states = $la->squeeze($states,axis:-1);
        if($states->dtype()!==NDArray::int32) {
            $states = $la->astype($states,NDArray::int32);
        }
        //return $this->table[$state];
        //$batches = $states->shape()[0];
        //if($size!=1) {
        //    throw new InvalidArgumentException('size of state detail must be 1');
        //}
        //$values = $la->alloc([$count,$size]);
        //foreach ($states as $key => $state) {
        //    $i = $state[0];
        //    $la->copy($this->table[R($i,$i+1)],$values[$key]);
        //}
        //$states = $la->squeeze($states,axis:-1);
        //$values = $la->gather($this->table,$states,$axis=null);
        // values(batches,numActions) = gather(table(numStates,numActions), states(batches))
        $actionValues = $la->gatherb($this->table,$states);
        //echo "table=";var_dump($this->table->toArray());
        //echo "states=";var_dump($states->toArray());
        //echo "actionValues=";var_dump($actionValues->toArray());
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
        //    //$la->masking($masks,$actionValues,fill:-INF);
        //    $la->masking($masks,$actionValues,fill:NAN);
        //    //$la->nan2num($values,-INF);
        //    //$la->add($masks,$values);   // 
        //}
        //echo "MaskedActionValues=";var_dump($actionValues->toArray());
        return $actionValues;
    }

    ///**
    //* $states       : (batches)          typeof int32
    //* $probablities : (batches,numActions) typeof float32
    //*/
    //public function sample(NDArray $states) : NDArray
    //{
    //    $la = $this->la;
    //    //$states = $la->squeeze($states,axis:-1);
    //    //$probs = $la->gather($this->rulesProbs,$states,$axis=null);
    //    $probablities = $la->gatherb($this->rulesProbs,$states); // (batches,numActions)
    //    //$actions = $this->randomCategorical($probs,1);
    //    return $probablities;
    //}
    
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

    /**
     * Action probabilities that do not include NaN
     * 
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

    public function fileExists(string $filename) : bool
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        return file_exists($filename);
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        $this->table->setPortableSerializeMode($mode);
    }

    public function saveWeightsToFile($filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $dump = serialize($this->table);
        file_put_contents($filename, $dump);
    }

    public function loadWeightsFromFile($filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $dump = file_get_contents($filename);
        $q = unserialize($dump);
        $this->table = $q;
    }
}
