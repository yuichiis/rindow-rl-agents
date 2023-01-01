<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\QPolicy;

class Greedy extends AbstractPolicy
{
    protected $qPolicy;

    public function __construct(
        $la)
    {
        parent::__construct($la);
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
        $qValues = $qPolicy->getQValues($state);
        $action = $la->reduceArgMax($qValues,$axis=-1);
        $action = $la->expandDims($action,$axis=-1);
        return $action;
    }
}
