<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Policy;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class PolicyGradient extends AbstractAgent
{
    protected $ones;
    protected $initialPolicy;
    protected $eta;
    protected $p;

    public function __construct($la, NDArray $policy,$eta,$mo=null)
    {
        $this->eta = $eta;
        $this->initialPolicy = $la->copy($policy);
        parent::__construct($la, $policy,$mo);
        [$ns,$na] = $policy->shape();
        $this->ones = $la->ones($la->alloc([$na]));
    }

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
    public function action($observation)
    {
        $action = $this->randomChoice($this->thresholds[$observation]);
        return $action;
    }

    /**
    * @param Any $params
    */
    public function update($experience) : void
    {
        $la = $this->la;
        $ones = $this->ones;
        $shape = $this->policy->shape();
        $steps = $experience->size();
        $rewards = $la->zeros($la->alloc([$steps]));
        $discountedReturn = $la->zeros($la->alloc([$steps]));

        $history = $experience->recently($steps);

        $ns =  $la->zeros($la->alloc($shape));
        $nsa = $la->zeros($la->alloc($shape));

        $totalReward = 0;
        $i = $steps-1;
        $discount = 0;
        $history = array_reverse($history);
        foreach ($history as $transition) {
            [$observation,$action,$nextObs,$reward,$done,$info] = $transition;
            $discount *= $gamma;
            $discount += $reward;
            $discountedReturn[$i] += $discount;
            baseline_value = value.predict(transition.state)
            advantage = $discountedReturn[$i] - baseline_value
            value.update(transition.state, total_return)
            policy.update(transition.state, advantage, transition.action)
        }

        // th(s,a) = th(s,a) + eta * (N(s,a)+P(s,a)*N(s))*R/T
        $totalStep = count($history); // T
        $th = $this->policy;      // th
        $p = $this->p;            // P
        $eta = $this->eta;        // eta
        $delta = $la->scal($eta*$totalReward/$totalStep,$la->axpy($nsa,$la->multiply($p,$ns)));
        $la->axpy($delta,$this->policy);
        $this->p = $this->generateProbabilities($this->policy);
        $this->thresholds = $this->generateThresholds($this->p);
    }
}
