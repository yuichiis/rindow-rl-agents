<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Metrics;
use Rindow\RL\Agents\Util\Metrics as MetricsImpl;
use InvalidArgumentException;
use LogicException;

abstract class AbstractAgent implements Agent
{
    abstract protected function estimator() : Estimator;

    protected object $la;
    protected ?Policy $policy;
    protected ?string $stateField;
    protected mixed $customRewardFunction=null;
    protected mixed $customStateFunction=null;
    protected ?Metrics $metrics=null;

    public function __construct(
        object $la,
        ?Policy $policy=null,
        ?string $stateField=null,
        )
    {
        $this->la = $la;
        $this->policy = $policy;
        $this->stateField = $stateField;
        $this->metrics = new MetricsImpl();
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

    public function resetData() : void
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
        NDArray|array $states,
        NDArray $action,
        NDArray|array $nextStates,
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
        NDArray|array $states,
        bool $done,
        bool $truncated,
        ?array $info,
        ) : NDArray|array
    {
        $func = $this->customStateFunction;
        if($func===null) {
            return $states;
        }
        return $func($env,$states,$done,$truncated,$info);
    }

    public function standardize(
        NDArray $x,         // (rolloutSteps)
        ?bool $ddof=null
        ) : NDArray
    {
        $ddof ??= false;

        $la = $this->la;

        // baseline
        $mean = $la->reduceMean($x,axis:0);     // ()
        $baseX = $la->add($mean,$la->copy($x),alpha:-1.0);  // (rolloutSteps)

        // std
        if($ddof) {
            $n = $x->size()-1;
        } else {
            $n = $x->size();
        }
        $variance = $la->scal(1/$n, $la->reduceSum($la->square($la->copy($baseX)),axis:0)); // ()
        $stdDev = $la->sqrt($variance); // ()

        // standardize
        $result = $la->multiply($la->reciprocal($stdDev,beta:1e-8),$baseX); // (rolloutSteps)
        return $result; // (rolloutSteps)
    }

    protected function log_prob_entropy_categorical(
        NDArray $logits,    // (batchsize,numActions) : float32
        NDArray $actions,   // (batchSize) : int32
    ) : array
    {
        $g = $this->g;
        $log_probs_all = $g->logSoftmax($logits); // (batchsize,numActions) : float32
        $selected_log_probs = $g->gather($log_probs_all, $actions, batchDims:1); // (batchsize) : float32

        $probs = $g->softmax($logits);  // (batchsize,numActions)
        $entropy = $g->scale(-1,$g->reduceSum($g->mul($probs, $log_probs_all), axis:1));

        return [$selected_log_probs, $entropy];
    }

    protected function log_prob_categorical(
        NDArray $logits,    // (batchsize,numActions) : float32
        NDArray $actions,   // (batchSize) : int32
    ) : NDArray
    {
        $g = $this->g;
        $la = $this->la;
        $log_probs_all = $g->logSoftmax($logits); // (batchsize,numActions) : float32
        $selected_log_probs = $g->gather($log_probs_all, $actions, batchDims:1); // (batchsize) : float32

        return $selected_log_probs; //, $entropy;
    }

    /**
     *  Args:
     *      mean (tf.Tensor):    平均
     *      logStd (tf.Tensor):  Logされた標準偏差
     *      value (tf.Tensor):   確率を計算したい値
     *  Returns:
     *      tuple[tf.Tensor, tf.Tensor]: (log_prob, entropy)
     */
    protected function log_prob_entropy_continuous(
        NDArray $mean,      // (batchSize,numActions)
        NDArray $logStd,    // (numActions) または (batchSize,numActions)
        NDArray $value,     // (batchSize,numActions)
    ) : array
    {
        $g = $this->g;
        // log_prob =
        //      -0.5 * tf.square((actions - mu) / tf.math.exp(log_std))
        //      -log_std
        //      -0.5 * np.log(2.0 * np.pi))
        $stableStd = $g->add($g->exp($logStd),$g->constant(1e-8));
        $logProb = $g->sub(
            $g->sub(
                $g->scale(-0.5,$g->square($g->div($g->sub($value,$mean),$stableStd))),
                $g->log($stableStd)
            ),
            $g->constant(0.5 * log(2.0 * pi()))
        );
        $logProb = $g->reduceSum($logProb, axis:1, keepdims:true);
        $entropy = $g->add($g->constant(0.5 + 0.5*log(2*pi())), $g->log($stableStd));
        $entropy = $g->add($g->zerosLike($mean),$entropy); // 他のテンソルとの互換性のため

        return [$logProb, $entropy]; // logProb=(batchsize,numActions), entropy=(1,numActions)
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

    protected function extractMask(NDArray|array $obs) : ?NDArray
    {
        $la = $this->la;
        if($obs instanceof NDArray) {
            return null;
        }
        if(!isset($obs['actionMask'])) {
            return null;
        }
        $mask = $obs['actionMask'];
        if(!($mask instanceof NDArray)) {
            throw new InvalidArgumentException("actionMask must NDArray");
        }
        return $mask;
    }

    protected function extractState(NDArray|array $obs) : NDArray
    {
        $la = $this->la;
        if($obs instanceof NDArray) {
            return $obs;
        }
        if($this->stateField===null) {
            throw new InvalidArgumentException("No valid state field name was specified.");
        }
        $stateField = $this->stateField;
        if(!isset($obs[$stateField])) {
            throw new InvalidArgumentException("The $stateField field is missing in observation.");
        }
        $state = $obs[$stateField];
        if(!($state instanceof NDArray)) {
            throw new InvalidArgumentException("The $stateField field must be NDArray.");
        }
        return $state;
    }

    protected function extractStateList(array $obsList) : array
    {
        if(count($obsList)===0) {
            return [];
        }
        if(!array_is_list($obsList)) {
            throw new InvalidArgumentException("observation list must be list array.");
        }
        if($obsList[0] instanceof NDArray) {
            return $obsList;
        }
        $stateList = [];
        foreach($obsList as $obs) {
            $stateList[] = $this->extractState($obs);
        }
        return $stateList;
    }

    protected function extractMaskList(array $obsList) : ?array
    {
        if(count($obsList)===0) {
            return [];
        }
        if(!array_is_list($obsList)) {
            throw new InvalidArgumentException("observation list must be list array.");
        }
        if($obsList[0] instanceof NDArray) {
            return null;
        }
        if(!isset($obsList[0]['actionMask'])) {
            return null;
        }
        $stateList = [];
        foreach($obsList as $obs) {
            $maskList[] = $this->extractMask($obs);
        }
        return $maskList;
    }

    public function reset(Env $env) : array
    {
        [$states,$info] = $env->reset();
        $states = $this->customState($env,$states,false,false,$info);
        return [$states,$info];
    }

    public function step(Env $env, int $episodeSteps, NDArray|array $states, ?array $info=null) : array
    {
        $la = $this->la;
        $action = $this->action($states,training:false,info:$info);
        [$nextStates,$reward,$done,$truncated,$info] = $env->step($action);
        $nextStates = $this->customState($env,$nextStates,$done,$truncated,$info);
        $reward = $this->customReward($env,$episodeSteps,$states,$action,$nextStates,$reward,$done,$truncated,$info);
        return [$nextStates,$reward,$done,$truncated,$info];
    }

    public function collect(
        Env $env,
        ReplayBuffer $experience,
        int $episodeSteps,
        NDArray|array $states,
        ?array $info,
        ) : array
    {
        $la = $this->la;
        $actions = $this->action($states,training:true,info:$info);
        [$nextState,$reward,$done,$truncated,$info] = $env->step($actions);
        $nextState = $this->customState($env,$nextState,$done,$truncated,$info);
        $reward = $this->customReward($env,$episodeSteps,$states,$actions,$nextState,$reward,$done,$truncated,$info);
        $experience->add([$states,$actions,$nextState,$reward,$done,$truncated,$info]);
        return [$nextState,$reward,$done,$truncated,$info];
    }

    /**
     * obs  : (batches, ...statesDims)
     * actions : (batches, ...ActionsDims)
     */
    public function action(NDArray|array $obs, ?bool $training=null, ?array $info=null, ?bool $parallel=null) : NDArray
    {
        $la = $this->la;
        $training ??= false;
        $info ??= [];
        $parallel ??= false;
        //[$states,$parallel,$isScalar] = $this->atleast2d($states);
        
        $masks = null;
        if($parallel) {
            //$states = $la->stack($states);
            if(!is_array($obs)) {
                throw new InvalidArgumentException("obs must be array when parallel=true.");
            }
            $states = [];
            $masks = [];
            $hasMask = false;
            $noMask = false;
            foreach($obs as $idx => $oneobs) {
                $states[] = $this->extractState($oneobs);
                $mask = $this->extractMask($oneobs);
                if($mask!==null) {
                    $hasMask = true;
                } else {
                    $noMask = true;
                }
                $masks[] = $mask;
            }
            if($hasMask && $noMask) {
                throw new InvalidArgumentException("Masks must be worn or not worn uniformly.");
            }
            $states = $la->stack($states,axis:0);
            if($hasMask) {
                $masks = $la->stack($masks,axis:0);
            } else {
                $masks = null;
            }
        } else {
            $states = $this->extractState($obs);
            $masks = $this->extractMask($obs);
            if($states->ndim()<1) {
                $shape = $la->shapeToString($states->shape());
                throw new InvalidArgumentException("shape of states must be greater than 1D. $shape given.");
            }
            $states = $la->expandDims($states,axis:0);
            if($masks!==null) {
                $masks = $la->expandDims($masks,axis:0);
            }
        }
        if($states->ndim()<2) {
            $shape = $la->shapeToString($states->shape());
            throw new InvalidArgumentException("shape of states must be greater than 1D or array of them. $shape given.");
        }

        // NDArray $states  : (batches,stateDims ) typeof int32 or float32
        // NDArray $actions : (batches) typeof int32 or (batches,numActions) typeof float32
        $actions = $this->policy->actions($this->estimator(),$states,training:$training,masks:$masks);
        if(!$parallel) {
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
