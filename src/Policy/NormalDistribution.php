<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator;

class NormalDistribution extends AbstractPolicy
{
    protected ?NDArray $min;
    protected ?NDArray $max;

    public function __construct(
        object $la,
        ?NDArray $min=null,
        ?NDArray $max=null,
        )
    {
        parent::__construct($la);
        $this->min = $min;
        $this->max = $max;
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

        $actions = $estimator->getActionValues($states);
        if($training) {
            $logStd = $estimator->getLogStd();
            $actions = $this->calcNormalDistSampled($actions,$logStd);
        }

        if($this->min!==null) {
            $actions = $la->maximum($la->copy($actions),$this->min);
        }
        if($this->max!==null) {
            $actions = $la->minimum($la->copy($actions),$this->max);
        }
        return $actions;
    }
}
