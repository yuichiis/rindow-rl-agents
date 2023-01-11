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
    * @param NDArray<any> $states
    * @return NDArray<int> $actions
    */
    public function action(QPolicy $qPolicy, NDArray $states, bool $training) : NDArray
    {
        return $this->calcMaxValueActions($qPolicy, $states);
    }
}
