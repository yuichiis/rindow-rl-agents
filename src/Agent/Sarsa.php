<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network\QTable;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Sarsa extends AbstractAgent
{
    protected $la;
    protected $qTable;
    protected $policy;
    protected $eta;
    protected $gamma;
    protected $threshold;

    public function __construct($la,
        QTable $qTable, Policy $policy,
        $eta,$gamma,$mo=null)
    {
        $this->la = $la;
        $this->qTable = $qTable;
        $this->policy = $policy;
        $this->eta = $eta;
        $this->gamma = $gamma;
    }

    public function initialize() // : Operation
    {
        //$la = $this->la;
        //$this->q = $la->randomUniform($this->policy->shape(),0,1);
        //$la->multiply($this->policy,$this->q);
        $this->qTable->initialize();
        $this->policy->initialize();
    }

    public function isStepUpdate() : bool
    {
        return true;
    }

    public function subStepLength() : int
    {
        return 2;
    }

    public function startEpisode(int $episode) : void
    {}

    public function endEpisode(int $episode) : void
    {}

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($observation,bool $training)
    {
        $la = $this->la;
        //if($training) {
            $action = $this->policy->action($observation,$this->elapsedTime);
        //} else {
        //    $qValues = $this->qTable->getQValues($observation);
        //    $action = $la->imax($qValues);
        //}
        return $action;
    }

    public function getQValue($observation) : float
    {
        $qValues = $this->qTable->getQValues($observation);
        $q = $this->la->max($qValues);
        return $q;
    }

    protected function getHistory($experience)
    {
        return $experience->recently(2);
    }

    protected function tdError(
        NDArray $q,NDArray $nextValues,float $reward,$history) : NDArray
    {
        $la = $this->la;
        [$dmy0,$nextAction,$dmy1,$dmy2,$dmy3,$dmy4] = $history[1];
        // R(t+1)+gamma*Q(s(t+1),a(t+1))-Q(s(t),a(t))
        $nextQ = $nextValues[[$nextAction,$nextAction]];
        $td = $la->axpy($q,$la->increment(
            $la->scal($this->gamma,$la->copy($nextQ)),$reward),
            -1.0);
        return $td;
    }

    /**
    * @param Any $params
    * @return Any $action
    */
    public function update($experience) : float
    {
        $la = $this->la;

        $history = $this->getHistory($experience);
        [$observation,$action,$nextObs,$reward,$done,$info] = $history[0];
        $table = $this->qTable->table();
        $q = $table[$observation][[$action,$action]];
        if($done) {
            // if done ($nextAction belongs to the next episode)
            // Q(s(t),a(t)) =
            //    Q(s(t),a(t))+eta*(*R(t+1)-Q(s(t),a(t))
            $tdError = $la->increment($la->copy($q),$reward,-1.0);
            $la->axpy($tdError,$q,$this->eta);
            $error = $la->nrm2($tdError);
        } else {
            // if continued
            // Q(s(t),a(t)) =
            //    Q(s(t),a(t))+eta*TDERROR
            // TDERROR = R(t+1) + gamma*Q(s(t+1),a(t+1)) - Q(s(t),a(t))
            $nextValues = $table[$nextObs];
            $tdError = $this->tdError($q,$nextValues,$reward,$history);
            $la->axpy($tdError,$q,$this->eta);
            $error = $la->nrm2($tdError);
        }
        return $error;
    }
}
