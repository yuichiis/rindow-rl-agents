<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\QPolicy;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

class UCB1 extends AbstractAgent
{
    protected $numActions;
    protected $numTrials;
    protected $numSuccesses;
    protected $values;
    protected $customRewardFunction;
    protected $step;
    protected $mo;

    public function __construct($la,
        QPolicy $qpolicy,
        $mo=null)
    {
        parent::__construct($la,null,null);
        $this->mo = $mo;
        $this->numActions = $qpolicy->numActions();
        $this->initialize();
    }

    public function policy()
    {
        return null;
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

    public function maxQValue(mixed $observation) : float
    {
        $q = $this->la->max($this->values);
        return $q;
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action(mixed $observation,bool $training) : mixed
    {
        $la = $this->la;
        if($training) {
            $i = $la->imin($this->numTrials);
            if($this->numTrials[$i]==0.0) {
                //echo "*".$i;
                return $i;
            }
            $action = $la->imax($this->values);
            //echo "#".$action;
        } else {
            $action = $la->imax($this->values);
            //echo "@".$action;
        }
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
