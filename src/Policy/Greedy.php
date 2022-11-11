<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\QPolicy;

class Greedy extends AbstractPolicy
{
    protected $qPolicy;

    public function __construct(
        $la, QPolicy $qPolicy)
    {
        parent::__construct($la);
        $this->qPolicy = $qPolicy;
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
        $qValues = $this->qPolicy->getQValues($state);
        return $this->la->imax($qValues);
    }
}
