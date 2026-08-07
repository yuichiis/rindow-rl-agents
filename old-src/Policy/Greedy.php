<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator;

class Greedy extends AbstractPolicy
{
    public function __construct(object $la)
    {
        parent::__construct($la);
    }

    public function initialize() : void
    {
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    /**
    * param  NDArray $states  : (batches,...StateDims) typeof int32 or float32
    * return NDArray $actions : (batches) typeof int32
    */
    public function actions(Estimator $estimator, NDArray $states, bool $training, ?NDArray $masks) : NDArray
    {
        return $this->calcMaxValueActions($estimator, $states, $masks);
    }
}
