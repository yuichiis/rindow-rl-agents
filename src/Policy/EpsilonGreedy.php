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
    * @param Any $states
    * @return Any $action
    */
    public function action(QPolicy $qPolicy, NDArray $state, bool $training) : NDArray
    {
        $la = $this->la;
        if($training && $this->threshold > mt_rand()) {
            $action = $qPolicy->sample($state);
        } else {
            $qValues = $qPolicy->getQValues($state);
            //$action = $this->la->imax($qValues);
            $action = $this->la->reduceArgMax($qValues,$axis=-1);
            $action = $la->expandDims($action,$axis=-1);
        }
        return $action;
    }
}
