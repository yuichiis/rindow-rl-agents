<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;

class Boltzmann extends AbstractPolicy
{
    use Random;

    protected $qPolicy;
    protected $tau;
    protected $min;
    protected $max;
    protected $softmax;

    public function __construct(
        $la, QPolicy $qPolicy,
        float $tau=null, float $min=null, float $max=null, bool $softmax=null)
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
        $this->softmax = $softmax;
    }

    public function initialize()
    {
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($state,bool $training)
    {
        $la = $this->la;
        // get probabilities
        $qValues = $this->qPolicy->getQValues($state);
        if(!$training) {
            $action = $la->imax($qValues);
            return $action;
        }

        if($this->softmax) {
            $qValues = $la->expandDims($qValues,$axis=0);
            if($this->tau!=1.0) {
                $qValues = $la->pow($la->copy($qValues),$this->tau);
            }
            $probabilities = $la->softmax($qValues);
            $probabilities = $la->squeeze($probabilities);
        } else {
            // q ** tau / sum(q ** tau)
            $expValues = $la->minimum($la->maximum(
                $la->pow($la->copy($qValues),$this->tau),$this->min),$this->max);
            $sum = $la->sum($expValues);
            if($sum==0) {
                $probabilities = $la->fill(1/$expValues->size(),$la->alloc($expValues->shape()));
            } else {
                $probabilities = $la->scal(1/$sum,$expValues);
            }
            unset($expValues);
        }

        // random choice with probabilities
        $action = $this->randomChoice($probabilities);
        return $action;
    }
}
