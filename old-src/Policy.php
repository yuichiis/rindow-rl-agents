<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\EventManager;

/**
 *
 */
interface Policy
{
    /**
    * @return void
    */
    public function initialize() : void;

    /**
    * @return void
    */
    public function register(?EventManager $eventManager=null) : void;

    public function isContinuousActions() : bool;

    /**
    * @param NDArray  $values  : (batches,...ValueDims)  typeof int32 or float32
    * @return NDArray $actions : (batches,...ActionDims) typeof int32 or float32
    */
    public function actions(Estimator $estimator, NDArray $values, bool $training, ?NDArray $masks) : NDArray;
}
