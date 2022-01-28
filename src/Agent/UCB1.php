<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\QPolicy;
use InvalidArgumentException;

class UCB1 extends AbstractAgent
{
    protected $la;
    protected $numActions;
    protected $numTrials;
    protected $numSuccesses;
    protected $values;
    protected $customRewardFunction;
    protected $step;

    public function __construct($la,QPolicy $qpolicy)
    {
        $this->la = $la;
        $this->numActions = $qpolicy->numActions();
        $this->initialize();
    }

    public function initialize() // : Operation
    {
        $numActions = $this->numActions;

        $la = $this->la;
        $this->values = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
        $this->numSuccesses = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
        $this->numTrials = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
        $this->step = 0;
    }

    public function isStepUpdate() : bool
    {
        return false;
    }

    public function subStepLength() : int
    {
        return 1;
    }

    public function startEpisode(int $episode) : void
    {
    }

    public function endEpisode(int $episode) : void
    {
    }

    public function getQValue($observation)
    {
        $q = $this->la->max($this->values);
        return $q;
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($observation,$training=null)
    {
        $la = $this->la;
        if($training) {
            $i = $this->la->imin($this->numTrials);
            if($this->numTrials[$i]==0.0) {
                return $i;
            }
            $action = $la->imax($this->values);
        } else {
            $action = $la->imax($this->values);
        }
        return $action;
    }

    /**
    * @param Any $params
    * @return Any $action
    */
    public function update($experience) : void
    {
        $la = $this->la;
        [$observation,$action,$nextObs,$reward,$done,$info] = $experience->last();
        if($action<0 || $action>=$this->numActions) {
            throw new InvalidArgumentException('Invalid Action');
        }
        $n1 = $this->numTrials[[$action,$action]];
        $w = $this->numSuccesses[[$action,$action]];
        $la->increment($n1,1.0);
        $la->increment($w,$reward);
        if($this->la->min($this->numTrials)==0.0) {
            return;
        }

        $n = $this->numTrials;
        $w = $this->numSuccesses;

        // V = (W/N) + sqrt(2*log(t)/N)
        $rn = $la->reciprocal($la->copy($n));
        $this->values = $la->axpy(
            $la->multiply($rn,$la->copy($w)),
            $la->sqrt($la->scal(2*log($this->step+1),$la->copy($rn))));

        $this->step++;
    }
}
