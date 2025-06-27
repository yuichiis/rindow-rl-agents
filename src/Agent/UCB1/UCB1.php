<?php
namespace Rindow\RL\Agents\Agent\UCB1;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Agent\AbstractAgent;
use InvalidArgumentException;
use LogicException;
use function Rindow\Math\Matrix\R;

class UCB1 extends AbstractAgent
{
    protected int $numActions;
    protected NDArray $numTrials;
    protected NDArray $numSuccesses;
    protected NDArray $values;
    protected int $step;
    protected ?object $mo;
    protected Estimator $estimator;

    public function __construct(
        object $la,
        int $numActions,
        ?object $mo=null)
    {
        parent::__construct($la);
        //$this->estimator = $estimator;
        $this->numActions = $numActions;
        $this->mo = $mo;
        $this->initialize();
    }

    public function policy() : ?Policy
    {
        return null;
    }

    public function estimator() : Estimator
    {
        throw new LogicException("Unsuppored operation.");
    }

    public function initialize() : void // : Operation
    {
        $numActions = $this->numActions;

        $la = $this->la;
        $this->values = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
        $this->numSuccesses = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
        $this->numTrials = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
        $this->step = 0;
    }

    public function resetData()
    {
        $this->initialize();
    }

    public function isStepUpdate() : bool
    {
        return false;
    }

    public function subStepLength() : int
    {
        return 1;
    }

    //public function maxQValue(mixed $state) : float
    //{
    //    $q = $this->la->max($this->values);
    //    return $q;
    //}

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action(array|NDArray $state, ?bool $training=null, ?array $info=null) : NDArray
    {
        $la = $this->la;
        if($training) {
            $i = $la->imin($this->numTrials);
            if($this->numTrials[$i]==0.0) {
                //echo "*".$i;
                $action = $i;
            } else {
                $action = $la->imax($this->values);
            }
            //echo "#".$action;
        } else {
            $action = $la->imax($this->values);
            //echo "@".$action;
        }
        return $la->array($action,dtype:NDArray::int32);
    }

    /**
    * @param Any $params
    * @return Any $action
    */
    public function update(ReplayBuffer $experience) : float
    {
        $la = $this->la;
        [$state,$action,$nextState,$reward,$done,$truncated,$info] = $experience->last();
        $action = $la->scalar($action);
        if($action<0 || $action>=$this->numActions) {
            throw new InvalidArgumentException('Invalid Action');
        }
        $n1 = $this->numTrials[R($action,$action+1)];
        $w = $this->numSuccesses[R($action,$action+1)];
        $la->increment($n1,1.0);
        $la->increment($w,$reward);
        if($this->la->min($this->numTrials)==0.0) {
            //echo "\n".$this->mo->toString($this->values,'%4.4f',true);
            return 0.0;
        }

        $n = $this->numTrials;
        $w = $this->numSuccesses;

        // V = (W/N) + sqrt(2*log(t)/N)
        $rn = $la->reciprocal($la->copy($n));
        $this->values = $la->axpy(
            $la->multiply($rn,$la->copy($w)),
            $la->sqrt($la->scal(2*log($this->step+1),$la->copy($rn))));

        $this->step++;
        //echo "\n".$this->mo->toString($this->values,'%4.4f',true);
        return 0.0;
    }

    public function fileExists(string $filename) : bool
    {
        return $this->estimator->fileExists($filename);
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        $this->estimator->setPortableSerializeMode($mode);
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $this->estimator->saveWeightsToFile($filename);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $this->estimator->loadWeightsFromFile($filename);
    }
}
