<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class PolicyGradient extends AbstractAgent
{
    use Random;

    const MODEL_FILENAME = '%s.model';

    protected $ones;
    protected $initialPolicy;
    protected $eta;
    protected $p;
    public $ptable;
    protected $mo;

    public function __construct($la,
        NDArray $rules, float $eta,
        EventManager $eventManager=null,
        $mo=null)
    {
        $table = $this->buildNetwork($la,$rules);
        $policy = $this->buildPolicy($la,$table);
        parent::__construct($la,$policy,$eventManager);
        $this->eta = $eta;
        $this->mo = $mo; // for debug
        $this->ptable = $table;
        $this->initialPolicy = $la->copy($rules);
        [$ns,$na] = $rules->shape();
        $this->ones = $la->ones($la->alloc([$na]));
        $this->initialize();
    }

    protected function buildNetwork($la,$rules)
    {
        $table = new PolicyTable($la,$rules);
        return $table;
    }

    protected function buildPolicy($la,$network)
    {
        $policy = new Boltzmann($la,fromLogits:true);
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

    public function initialize() // : Operation
    {
        //$la = $this->la;
        //$la->copy($this->initialPolicy,$this->policy);
        //$this->p = $this->generateProbabilities($this->policy);
        //$this->thresholds = $this->generateThresholds($this->p);
        $this->policy->initialize();
    }

    public function resetData()
    {
        $this->ptable->initialize();
    }

    protected function policyTable() : QPolicy
    {
        return $this->ptable;
    }

    ///**
    //* @param Any $states
    //* @return Any $action
    //*/
    //public function action($observation,bool $training)
    //{
    //    if($training) {
    //        $action = $this->randomChoice($this->thresholds[$observation], isThresholds:true);
    //    } else {
    //        $action = $this->la->imax($this->p[$observation]);
    //    }
    //    return $action;
    //}

    public function maxQValue(mixed $observation) : float
    {
        $p = $this->la->max($this->p[$observation]);
        return $p;
    }

    /**
    * @param Any $params
    */
    public function update($experience) : float
    {
        $la = $this->la;
        $ones = $this->ones;
        $shape = $this->policy->shape();
        $ns =  $la->zeros($la->alloc($shape));
        $nsa = $la->zeros($la->alloc($shape));
        $history = $experience->recently($experience->size());

        $totalReward = 0;
        foreach ($history as $transition) {
            [$observation,$action,$nextObs,$reward,$done,$info] = $transition;
            $la->increment($nsa[$observation][[$action,$action]],1.0);
            $la->axpy($ones,$ns[$observation]);
        }

        // th(s,a) = th(s,a) + eta * (N(s,a)+P(s,a)*N(s))/T
        $totalStep = count($history); // T

        $th = $this->ptable->table(); // th
        $p = $this->p;            // P
        $eta = $this->eta;        // eta
        $delta = $la->scal($eta/$totalStep,$la->axpy($nsa,$la->multiply($p,$ns)));
        $la->axpy($delta,$th);

        $this->p = $this->generateProbabilities($this->ptable->table());
        //$this->thresholds = $this->generateThresholds($this->p);
        $experience->clear();
        
        return $la->nrm2($delta);
    }

    public function fileExists(string $filename) : bool
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        return file_exists($filename);
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        $this->policy->setPortableSerializeMode($mode);
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $dump = serialize($this->policy);
        file_put_contents($filename, $dump);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $dump = file_get_contents($filename);
        $policy = unserialize($dump);
        $la->copy($policy,$this->policy);
        //$this->p = $this->generateProbabilities($this->policy);
        //$this->thresholds = $this->generateThresholds($this->p);
    }
}
