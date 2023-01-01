<?php
namespace Rindow\RL\Agents\Network;

use Interop\Polite\Math\Matrix\NDArray;

class Probabilities extends QTable
{
    protected $probs;
    protected $onesProb;

    public function __construct($la, NDArray $probs)
    {
        $this->la = $la;
        $this->probs = $probs;
        [$obsSize,$numActions] = $probs->shape();
        $this->obsSize = $obsSize;
        $this->numActions = $numActions;
        $p = $la->alloc([1,$numActions]);
        $this->onesProb = $la->ones($p);
        $this->initialize();
    }

    public function initialize() // : Operation
    {
        $la = $this->la;
        //$this->thresholds = $this->generateThresholds($this->probs);
        $this->q = $la->randomUniform($this->probs->shape(),0,1);
    }

    public function sample(NDArray $state) : NDArray
    {
        //$action = mt_rand(0,$this->numActions-1);
        $action = $this->randomCategorical($this->onesProb,count($state));
        $action = $la->expandDims($la->squeeze($action),$axis=1);
        return $action;
    }
}
