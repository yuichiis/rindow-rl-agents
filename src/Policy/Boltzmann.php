<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;

class Boltzmann implements Policy
{
    use Random;

    protected $la;
    protected $qPolicy;
    protected $tau;
    protected $min;
    protected $max;

    public function __construct(
        $la, QPolicy $qPolicy,
        float $tau=null, float $min=null, float $max=null)
    {
        if($tau===null) {
            $tau=1.0;
        }
        if($min===null) {
            $min=-500.0;
        }
        if($max===null) {
            $max=500.0;
        }
        $this->la = $la;
        $this->qPolicy = $qPolicy;
        $this->tau = $tau;
        $this->min = $min;
        $this->max = $max;
    }

    public function initialize()
    {
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($state,int $time=null)
    {
        $la = $this->la;
        // get probabilities
        $qValues = $this->qPolicy->getQValues($state);
        $expValues = $la->exp($la->minimum($la->maximum(
            $la->scal(1/$this->tau,$la->copy($qValues)),$this->min),$this->max));
        $probabilities = $la->scal(1/$la->sum($expValues),$expValues);
        unset($expValues);

        // random choice with probabilities
        $probabilities = $la->expandDims($probabilities,$axis=0);
        $thresholds = $this->generateThresholds($probabilities);
        $thresholds = $la->squeeze($thresholds,$axis=0);
        $action = $this->randomChoice($thresholds);
        return $action;
    }
}
