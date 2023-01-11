<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\EventManager;

abstract class AbstractPolicy implements Policy
{
    protected $la;

    public function __construct($la)
    {
        $this->la = $la;
    }

    public function register(EventManager $eventManager=null) : void
    {
    }

    /**
    * @param NDArray<any> $states
    * @return NDArray<int> $actions
    */
    protected function calcMaxValueActions(QPolicy $qPolicy, NDArray $states) : NDArray
    {
        $la = $this->la;
        $qValues = $qPolicy->getQValues($states);
        $actions = $la->reduceArgMax($qValues,$axis=-1);
        $actions = $la->expandDims($actions,$axis=-1);
        return $actions;
    }
}