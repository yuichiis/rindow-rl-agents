<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Metrics;
use Rindow\RL\Agents\ReplayBuffer as ReplayBufferInterface;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use InvalidArgumentException;
use LogicException;

abstract class AbstractAgent implements Agent
{
    abstract protected function estimator() : Estimator;

    protected object $la;
    protected ?Policy $policy;
    protected mixed $customRewardFunction=null;
    protected mixed $customStateFunction=null;
    protected ?Metrics $metrics;

    public function __construct(
        object $la,
        ?Policy $policy=null,
        )
    {
        $this->la = $la;
        $this->policy = $policy;
    }

    public function register(?EventManager $eventManager=null) : void
    {
        $policy = $this->policy;
        if($policy instanceof Policy) {
            $policy->register($eventManager);
        }
    }

    public function policy() : ?Policy
    {
        return $this->policy;
    }

    public function setMetrics(Metrics $metrics) : void
    {
        $this->metrics = $metrics;
    }

    public function metrics() : Metrics
    {
        return $this->metrics;
    }

    public function resetData()
    {
        throw new LogicException('unsuported operation');
    }

    public function setCustomRewardFunction(callable $func) : void
    {
        $this->customRewardFunction = $func;
    }

    public function setCustomStateFunction(callable $func) : void
    {
        $this->customStateFunction = $func;
    }

    protected function customReward(
        Env $env,
        int $stepCount,
        NDArray $states,
        NDArray $action,
        NDArray $nextStates,
        float $reward,
        bool $done,
        bool $truncated,
        ?array $info,
        ) : float
    {
        $func = $this->customRewardFunction;
        if($func===null) {
            return $reward;
        }
        return $func($env,$stepCount,$states,$action,$nextStates,$reward,$done,$truncated,$info);
    }

    protected function customState(
        Env $env,
        NDArray $states,
        bool $done,
        bool $truncated,
        ?array $info,
        ) : NDArray
    {
        $func = $this->customStateFunction;
        if($func===null) {
            return $states;
        }
        return $func($env,$states,$done,$truncated,$info);
    }

    public function atleast2d(mixed $states) : NDArray
    {
        $la = $this->la;
        if($states instanceof NDArray) {
            if($states->ndim()===0) {
                return $states->reshape([1,1]);
            }
            return $la->expandDims($states,$axis=0);
        } elseif(is_numeric($states)) {
            return $la->array([[$states]]);
        } if(!is_array($states)) {
            throw new InvalidArgumentException('states must be NDarray or numeric or array.');
        }
        if($states[0] instanceof NDArray) {
            $states = $la->stack($states,$axis=0);
        } else {
            $states = $la->expandDims($la->array($states),$axis=0);
        }
        return $states;
    }

    protected function extractMasks(?array $infos) : ?NDArray
    {
        $la = $this->la;
        $masks = null;
        if($infos!=null) {
            $masks = [];
            foreach($infos as $inf) {
                if(!isset($inf['validActions'])) {
                    $masks = null;
                    break;
                    }
                $masks[] = $inf['validActions'] ?? null;
            }
            if($masks!==null) {
                if(count($masks)>0) {
                    $masks = $la->stack($masks);
                } else {
                    $masks = null;
                }
            }
        }
        return $masks;
    }

    public function reset(Env $env) : array
    {
        [$states,$info] = $env->reset();
        $states = $this->customState($env,$states,false,false,$info);
        return [$states,$info];
    }

    public function step(Env $env, int $episodeSteps, NDArray $states, ?array $info=null) : array
    {
        $la = $this->la;
        $action = $this->action($states,training:false,info:$info);
        [$nextStates,$reward,$done,$truncated,$info] = $env->step($action);
        $nextStates = $this->customState($env,$nextStates,$done,$truncated,$info);
        $reward = $this->customReward($env,$episodeSteps,$states,$action,$nextStates,$reward,$done,$truncated,$info);
        return [$nextStates,$reward,$done,$truncated,$info];
    }

    /**
     * states  : (batches, ...statesDims)
     * actions : (batches, ...ActionsDims)
     */
    public function action(array|NDArray $states, ?bool $training=null, ?array $info=null) : NDArray
    {
        $la = $this->la;
        $training ??= false;
        $info ??= [];
        //[$states,$isParallel,$isScalar] = $this->atleast2d($states);
        
        $masks = null;
        $isParallel = is_array($states);
        if($isParallel) {
            $states = $la->stack($states);
            $masks = $this->extractMasks($info);
        } else {
            if($states->ndim()<1) {
                $shape = $la->shapeToString($states->shape());
                throw new InvalidArgumentException("shape of states must be greater than 1D. $shape given.");
            }
            $states = $la->expandDims($states,axis:0);
            if($info!=null) {
                $masks = $info['validActions'] ?? null;
                if($masks!==null) {
                    $masks = $la->expandDims($masks,axis:0);
                }
            }
        }
        if($states->ndim()<2) {
            $shape = $la->shapeToString($states->shape());
            throw new InvalidArgumentException("shape of states must be greater than 1D or array of them. $shape given.");
        }

        // NDArray $states  : (batches,stateDims ) typeof int32 or float32
        // NDArray $actions : (batches) typeof int32 or (batches,numActions) typeof float32
        $actions = $this->policy->actions($this->estimator(),$states,training:$training,masks:$masks);
        if(!$isParallel) {
            $actions = $la->squeeze($actions,axis:0);
        }
        //echo "states(".implode(',',$states->shape()).")\n";
        //echo "action(".implode(',',$actions->shape()).")\n";
        return $actions;
    }

    //public function maxQValue(mixed $states) : float
    //{
    //    $la = $this->la;
    //    $states = $this->atleast2d($states);
    //    $qValues = $this->estimator()->getActionValues($states);
    //    $q = $la->max($qValues);
    //    return $q;
    //}

}
