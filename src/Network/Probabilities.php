<?php
namespace Rindow\RL\Agents\Network;

use Interop\Polite\Math\Matrix\NDArray;

class Probabilities extends QTable
{
    protected $probs;

    public function __construct($la, NDArray $probs)
    {
        $this->la = $la;
        $this->probs = $probs;
        [$obsSize,$numActions] = $probs->shape();
        $this->obsSize = $obsSize;
        $this->numActions = $numActions;
        $this->initialize();
    }

    public function initialize() // : Operation
    {
        $la = $this->la;
        $this->thresholds = $this->generateThresholds($this->probs);
        $this->q = $la->randomUniform($this->probs->shape(),0,1);
    }

    public function sample($state)
    {
        $action = mt_rand(0,$this->numActions-1);
        return $action;
    }
}
