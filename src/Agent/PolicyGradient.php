<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class PolicyGradient extends AbstractAgent
{
    use Random;

    protected $la;
    protected $ones;
    protected $initialPolicy;
    protected $eta;
    protected $p;

    public function __construct($la, NDArray $rules,$eta,$mo=null)
    {
        $this->la = $la;
        $this->eta = $eta;
        $this->initialPolicy = $la->copy($rules);
        [$ns,$na] = $rules->shape();
        $this->ones = $la->ones($la->alloc([$na]));
    }

    public function policy()
    {
        return null;
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
    {}

    public function endEpisode(int $episode) : void
    {}

    public function initialize() // : Operation
    {
        $la = $this->la;
        $this->policy = $la->copy($this->initialPolicy);
        $this->p = $this->generateProbabilities($this->policy);
        $this->thresholds = $this->generateThresholds($this->p);
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($observation,bool $training)
    {
        if($training) {
            $action = $this->randomChoice($this->thresholds[$observation]);
        } else {
            $action = $this->la->imax($this->p[$observation]);
        }
        return $action;
    }

    public function getQValue($observation) : float
    {
        $p = $this->la->max($this->p[$observation]);
        return $p;
    }

    /**
    * @param Any $params
    */
    public function update($experience) : void
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
            $totalReward += $reward;
        }

        // th(s,a) = th(s,a) + eta * (N(s,a)+P(s,a)*N(s))/T
        $totalStep = count($history); // T
        $th = $this->policy;      // th
        $p = $this->p;            // P
        $eta = $this->eta;        // eta
        $delta = $la->scal($eta*$totalReward/$totalStep,$la->axpy($nsa,$la->multiply($p,$ns)));
        $la->axpy($delta,$this->policy);
        $this->p = $this->generateProbabilities($this->policy);
        $this->thresholds = $this->generateThresholds($this->p);
        $experience->clear();
    }
}
