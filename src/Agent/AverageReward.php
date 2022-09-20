<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class AverageReward extends AbstractAgent
{
    protected $la;
    protected $qpolicy;
    protected $numActions;
    protected $numTrials;
    protected $values;
    protected $customRewardFunction;

    public function __construct($la, QPolicy $qpolicy, Policy $policy)
    {
        $this->la = $la;
        $this->qpolicy = $qpolicy;
        $this->numActions = $qpolicy->numActions();
        $this->setPolicy($policy);
        $this->initialize();
    }

    public function initialize() // : Operation
    {
        $la = $this->la;
        $numActions = $this->numActions;
        $this->values = $this->qpolicy->table();
        $la->zeros($this->values);
        $this->numTrials = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
        $this->policy->initialize();
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

    public function getQValue($observation) : float
    {
        $qValues = $this->qpolicy->getQValues($observation);
        $q = $this->la->max($qValues);
        return $q;
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($observation,bool $training)
    {
        //$la = $this->la;
        //if($training) {
            $action = $this->policy->action($observation,$training,$this->elapsedTime);
        //} else {
        //    $action = $la->imax($this->values);
        //}
        return $action;
    }

    /**
    * @param Any $params
    * @return Any $action
    */
    public function update($experience) : float
    {
        $la = $this->la;
        [$observation,$action,$nextObs,$reward,$done,$info] = $experience->last();
        if($action<0 || $action>=$this->numActions) {
            throw new InvalidArgumentException('Invalid Action');
        }
        $n = $this->numTrials[[$action,$action]];
        $v = $this->values[$observation][[$action,$action]];

        $la->increment($n,1.0);

        // V(t) = ((n-1)/n)*V(t-1) + 1/n*R(t)
        //      = ((1-1/n))*V(t-1) + 1/n*R(t)
        $la->multiply(
            $la->increment($la->reciprocal($la->copy($n),null,-1.0),1.0),
            $v);
        $la->axpy(
            $la->scal($reward,$la->reciprocal($la->copy($n))),
            $v);

        return 0.0;
    }

    public function fileExists(string $filename) : bool
    {
        return $this->qpolicy->fileExists($filename);
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        $this->qpolicy->setPortableSerializeMode($mode);
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $this->qpolicy->saveWeightsToFile($filename);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $this->qpolicy->loadWeightsFromFile($filename);
    }
}
