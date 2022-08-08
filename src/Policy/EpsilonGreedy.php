<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;

class EpsilonGreedy implements Policy
{
    protected $la;
    protected $qPolicy;
    protected $epsilon;
    protected $numActions;
    protected $threshold;

    public function __construct(
        $la,QPolicy $qPolicy,
        float $epsilon=null)
    {
        if($epsilon===null) {
            $epsilon = 0.1;
        }
        $this->la = $la;
        $this->qPolicy = $qPolicy;
        $this->epsilon = $epsilon;
        $this->numActions = $this->qPolicy->numActions();
        $this->threshold = (int)floor($epsilon * getrandmax());
    }

    public function initialize()
    {
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($state,bool $training,int $time=null)
    {
        if($training && $this->threshold > mt_rand()) {
            $action = mt_rand(0,$this->numActions-1);
        } else {
            $qValues = $this->qPolicy->getQValues($state);
            $action = $this->la->imax($qValues);
        }
        return $action;
    }
}
