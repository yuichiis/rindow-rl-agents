<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\QPolicy;

class EpsilonGreedy extends AbstractPolicy
{
    protected $qPolicy;
    protected $epsilon;
    protected $threshold;

    public function __construct(
        $la,
        float $epsilon=null)
    {
        if($epsilon===null) {
            $epsilon = 0.1;
        }
        parent::__construct($la);
        $this->epsilon = $epsilon;
        $this->threshold = (int)floor($epsilon * getrandmax());
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
        $la = $this->la;
        if(!$training) {
            return $this->calcMaxValueActions($qPolicy, $states);
        }

        if($this->threshold > mt_rand()) {
            $actions = $qPolicy->sample($states);
        } else {
            $actions = $this->calcMaxValueActions($qPolicy, $states);
        }
        return $actions;
    }
}
