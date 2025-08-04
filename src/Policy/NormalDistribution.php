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
        if(!$training) {
            return $estimator->getActionValues($states);
        }

        // get values
        $actionMeans = $estimator->getActionValues($states);
        $logStd = $estimator->getLogStd();
        $actions = $this->calcNormalDistSampled($actionMeans,$logStd);

        if($this->min) {
            $actions = $la->maximum($actions,$this->min);
        }
        if($this->max) {
            $actions = $la->minimum($actions,$this->max);
        }

        return $actions;
    }
}
