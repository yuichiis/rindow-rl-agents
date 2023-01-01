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
    protected $fromLogits;

    public function __construct(
        $la,
        float $tau=null, float $min=null, float $max=null, bool $fromLogits=null)
    {
        if($tau===null) {
            $tau=1.0;
        }
        if($min===null) {
            if($fromLogits) {
                $min=-500.0;
            } else {
                $min=0;
            }
        }
        if($max===null) {
            $max=500.0;
        }
        $this->la = $la;
        $this->tau = $tau;
        $this->min = $min;
        $this->max = $max;
        $this->fromLogits = $fromLogits;
    }

    public function initialize()
    {
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action(QPolicy $qPolicy, NDArray $state, bool $training) : NDArray
    {
        $la = $this->la;
        // get probabilities
        $qValues = $qPolicy->getQValues($state);
        if(!$training) {
            $action = $la->reduceArgMax($qValues,$axis=-1);
            $action = $la->expandDims($action,$axis=-1);
            return $action;
        }

        $qValues = $la->copy($qValues);
        if($this->tau!=1.0) {
            $qValues = $la->pow($qValues,$this->tau);
        }
        $qValues = $la->minimum($la->maximum($qValues,$this->min),$this->max);
        if($this->fromLogits) {
            $qValues = $la->exp($qValues);
        }
        $qValues = $la->nan2num($qValues);

        // random choice with probabilities
        $action = $this->randomCategorical($qValues,1);
        return $action;
    }
}
