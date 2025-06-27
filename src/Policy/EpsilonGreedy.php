<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator;

class EpsilonGreedy extends AbstractPolicy
{
    protected float $epsilon;
    protected int $threshold;

    public function __construct(
        object $la,
        ?float $epsilon=null)
    {
        $epsilon ??= 0.1;

        parent::__construct($la);
        $this->epsilon = $epsilon;
        $this->threshold = (int)floor($epsilon * getrandmax());
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    public function initialize() : void
    {
    }

    /**
    * param  NDArray $states  : (batches,...StateDims) typeof int32 or float32
    * return NDArray $actions : (batches) typeof int32
    */
    public function actions(Estimator $estimator, NDArray $states, bool $training, ?NDArray $masks) : NDArray
    {
        $la = $this->la;
        if(!$training) {
            return $this->calcMaxValueActions($estimator, $states, $masks);
        }

        if($this->threshold > mt_rand()) {
            $actions = $this->calcRandomActions($estimator, $states, $masks);
        } else {
            $actions = $this->calcMaxValueActions($estimator, $states, $masks);
        }
        return $actions;
    }
}
