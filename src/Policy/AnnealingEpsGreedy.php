<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;

class AnnealingEpsGreedy implements Policy
{
    protected $la;
    protected $qPolicy;
    protected $decayRate;
    protected $start;
    protected $stop;
    protected $numActions;
    protected $currentTime = 0;

    public function __construct(
        $la, QPolicy $qPolicy,
        float $start=null,float $stop=null,float $decayRate=null)
    {
        if($decayRate===null) {
            $decayRate = 0.01;
        }
        if($start===null) {
            $start = 0.9;
        }
        if($stop===null) {
            $stop = 0.1;
        }
        $this->la = $la;
        $this->qPolicy = $qPolicy;
        $this->decayRate = $decayRate;
        $this->start = $start;
        $this->stop = $stop;
        $this->numActions = $this->qPolicy->numActions();
        $this->initialize();
    }

    public function initialize()
    {
        $this->currentTime = 0;
    }

    public function getEpsilon(int $time=null)
    {
        if($time===null) {
            $time = $this->currentTime;
        }
        return $this->stop + ($this->start-$this->stop)*exp(-$this->decayRate*$time);
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($state,bool $training,$time=null)
    {
        $epsilon = $this->getEpsilon($time);
        $threshold = (int)floor($epsilon * getrandmax());
        $numActions = $this->numActions;
        if($training && $threshold > mt_rand()) {
            $action = $this->qPolicy->sample($state);
        } else {
            $qValues = $this->qPolicy->getQValues($state);
            $action = $this->la->imax($qValues);
        }
        if($training) {
            $this->currentTime++;
        }
        return $action;
    }
}
