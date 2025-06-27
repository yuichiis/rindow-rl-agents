<?php
namespace Rindow\RL\Agents\Agent\PolicyGradient;

use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Agent\AbstractAgent;
use Rindow\RL\Agents\Policy\Boltzmann;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

class PolicyGradient extends AbstractAgent
{
    protected NDArray $ones;
    //protected NDArray $mask;
    protected float $eta;
    protected PolicyTable $pTable;
    protected NDArray $prevProbs;
    protected ?object $mo;

    public function __construct(
        object $la,
        int $numStates,
        int $numActions,
        float $eta,
        ?PolicyTable $table=null,
        ?Policy $policy=null,
        ?EventManager $eventManager=null,
        ?object $mo=null,
        )
    {
        $table ??= $this->buildTable($la,$numStates, $numActions);
        $policy ??= $this->buildPolicy($la);
        parent::__construct($la,$policy,$eventManager);
        $this->eta = $eta;
        $this->mo = $mo; // for debug
        $this->pTable = $table;
        //$this->mask = $rules;
        //[$ns,$na] = $rules->shape();
        $this->ones = $la->ones($la->alloc([$numActions]));
        $this->resetData();
        $this->initialize();
    }

    protected function buildTable(
        object $la,
        int $numStates,
        int $numActions,
        ) : PolicyTable
    {
        $table = new PolicyTable($la,$numStates, $numActions);
        return $table;
    }

    protected function buildPolicy(object $la) : Policy
    {
        $policy = new Boltzmann($la);
        return $policy;
    }

    public function isStepUpdate() : bool
    {
        return false;
    }

    public function subStepLength() : int
    {
        return 1;
    }

    public function initialize() : void // : Operation
    {
        $this->policy->initialize();
    }

    public function resetData()
    {
        $la = $this->la;
        $this->pTable->initialize();
        // init prevProbs
        $logits = $la->ones($la->alloc($this->pTable->table()->shape()));
        //if($this->mask) {
        //    $la->masking($this->mask,$logits,fill:-INF);
        //}
        $this->prevProbs = $la->softmax($logits);

    }

    protected function estimator() : Estimator
    {
        return $this->pTable;
    }

    ///**
    //* @param Any $states
    //* @return Any $action
    //*/
    //public function action($state, ?bool $training=null,?array $info=null)
    //{
    //    if($training) {
    //        $action = $this->randomChoice($this->thresholds[$state], isThresholds:true);
    //    } else {
    //        $action = $this->la->imax($this->p[$state]);
    //    }
    //    return $action;
    //}

    //public function maxQValue(mixed $state) : float
    //{
    //    $p = $this->la->max($this->p[$state]);
    //    return $p;
    //}

    /**
    * @param Any $params
    */
    public function update(ReplayBuffer $experience) : float
    {
        $la = $this->la;
        $ones = $this->ones;
        $table = $this->estimator()->table();   // table
        $shape = $table->shape();
        $ns =  $la->zeros($la->alloc($shape));
        $nsa = $la->zeros($la->alloc($shape));
        $history = $experience->recently($experience->size());

        $totalReward = 0;
        foreach ($history as $transition) {
            [$state,$action,$nextState,$reward,$done,$truncated,$info] = $transition;
            if($state->shape()!==[1]) {
                throw new LogicException("Shape of State in replay buffer must be (1).".$la->shapeToString($state->shape()));
            }
            $stateNumber = $la->scalar($la->squeeze($state));
            $actionNumber = $la->scalar($la->squeeze($action));
            $la->increment($nsa[$stateNumber][R($actionNumber,$actionNumber+1)],1.0);
            $la->axpy($ones,$ns[$stateNumber]);
        }

        // th(s,a) = th(s,a) + eta * (N(s,a)+P(s,a)*N(s))/T
        $totalStep = count($history); // T

        $th = $table;                       // th
        $p = $this->prevProbs;              // P    // probabilities
        $eta = $this->eta;                  // eta
        $delta = $la->scal($eta/$totalStep,$la->axpy($nsa,$la->multiply($p,$ns)));
        $la->axpy($delta,$th);

        // from policy to probabileties
        $logits = $la->log($la->copy($table));
        //if($this->mask) {
        //    $la->masking($this->mask,$logits,fill:-INF);
        //}
        $this->prevProbs = $la->softmax($logits);

        $experience->clear();
        
        return $la->nrm2($delta);
    }

    public function fileExists(string $filename) : bool
    {
        return $this->estimator()->fileExists($filename);
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        $this->estimator()->setPortableSerializeMode($mode);
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $this->estimator()->saveWeightsToFile($filename);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $this->estimator()->loadWeightsFromFile($filename);
    }
}
