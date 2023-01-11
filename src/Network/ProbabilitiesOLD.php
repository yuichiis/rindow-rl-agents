<?php
namespace Rindow\RL\Agents\Network;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Probabilities extends QTable
{
    public function __construct($la, NDArray $probs)
    {
        $this->la = $la;
        [$numObs,$numActions] = $probs->shape();
        if($numObs!=1) {
            throw new InvalidArgumentException('Num of obs must be one');
        }
        $p = $la->alloc([1,$numActions]);
        $rules = $la->ones($p);
        parent::__construct($la, $rules);
    }

    //public function sample(NDArray $states) : NDArray
    //{
    //    //$actions = mt_rand(0,$this->numActions-1);
    //    $actions = $this->randomCategorical($this->onesProb,count($states));
    //    $actions = $la->expandDims($la->squeeze($actions),$axis=1);
    //    return $actions;
    //}
}
