<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\EventManager;
use InvalidArgumentException;
use LogicException;

abstract class AbstractAgent implements Agent
{
    abstract protected function policyTable() : QPolicy;

    protected $la;
    protected $policy;

    public function __construct(object $la,
        Policy|NDArray $policy=null, EventManager $eventManager=null)
    {
        $this->la = $la;
        $this->policy = $policy;
    }

    public function register(EventManager $eventManager=null) : void
    {
        $policy = $this->policy;
        if($policy instanceof Policy) {
            $policy->register($eventManager);
        }
    }

    public function policy()
    {
        return $this->policy;
    }

    public function resetData()
    {
        throw new LogicException('unsuported operation');
    }

    public function atleast2d(mixed $obs) : NDArray
    {
        $la = $this->la;
        if($obs instanceof NDArray) {
            return $la->expandDims($obs,$axis=0);
        } elseif(is_numeric($obs)) {
            return $la->array([[$obs]]);
        } if(!is_array($obs)) {
            throw new InvalidArgumentException('observations must be NDarray or numeric or array.');
        }
        if($obs[0] instanceof NDArray) {
            $obs = $la->stack($obs,$axis=0);
        } else {
            $obs = $la->expandDims($la->array($obs),$axis=0);
        }
        return $obs;
    }

    public function action(mixed $observation,bool $training) : mixed
    {
        $la = $this->la;
        $multi = is_array($observation);
        $isScalar = false;
        if(!$multi) {
            $isScalar = is_scalar($observation);
        }
        $observation = $this->atleast2d($observation);
        $action = $this->policy->action($this->policyTable(),$observation,$training);
        if(!$multi) {
            $action = $la->squeeze($action,$axis=0);
            if($action->shape()==[1]&&$isScalar) {
                $action = $la->squeeze($action,$axis=0);
                $action = $la->scalar($action);
            }
        }
        return $action;
    }

    public function maxQValue(mixed $observation) : float
    {
        $la = $this->la;
        $observation = $this->atleast2d($observation);
        $qValues = $this->policyTable()->getQValues($observation);
        $q = $la->max($qValues);
        return $q;
    }

}
