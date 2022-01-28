<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class QTable implements QPolicy
{
    use Random;
    protected $rules;
    protected $thresholds;
    protected $obsSize;
    protected $numActions;

    public function __construct($la, NDArray $rules)
    {
        $this->la = $la;
        $this->rules = $rules;
        [$obsSize,$numActions] = $rules->shape();
        $this->obsSize = $obsSize;
        $this->numActions = $numActions;
        $this->initialize();
    }

    public function initialize() // : Operation
    {
        $la = $this->la;
        // initialize legal actions probabilities
        $p = $this->generateProbabilities($this->rules);
        $this->thresholds = $this->generateThresholds($p);
        // initialize Q table
        $this->q = $la->randomUniform($this->rules->shape(),0,1);
        $la->multiply($this->rules,$this->q);
    }

    public function obsSize()
    {
        return $this->obsSize;
    }

    public function numActions() : int
    {
        return $this->numActions;
    }

    public function table()
    {
        return $this->q;
    }

    public function getQValues($state) : NDArray
    {
        if(!is_int($state)) {
            throw new InvalidArgumentException('state must be integer');
        }
        return $this->q[$state];
    }

    public function sample($state)
    {
        if(!is_int($state)) {
            throw new InvalidArgumentException('state must be integer. '.gettype($state).' given.');
        }
        $action = $this->randomChoice($this->thresholds[$state]);
        return $action;
    }
}
