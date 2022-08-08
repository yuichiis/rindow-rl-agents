<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;

class Greedy implements Policy
{
    protected $la;
    protected $qPolicy;

    public function __construct(
        $la, QPolicy $qPolicy)
    {
        $this->la = $la;
        $this->qPolicy = $qPolicy;
    }

    public function initialize()
    {
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($state,bool $training=null,int $time=null)
    {
        $qValues = $this->qPolicy->getQValues($state);
        return $this->la->imax($qValues);
    }
}
