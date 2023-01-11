<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\OUProcess;

class OUNoise extends AbstractPolicy
{
    protected $qPolicy;

    public function __construct(
        $la,
        NDArray $mean,
        NDArray $std_dev,
        NDArray $lower_bound,
        NDArray $upper_bound,
        float $theta=null,
        float $dt=null,
        NDArray $x_initial=null
        )
    {
        parent::__construct($la);
        $this->lower_bound = $lower_bound;
        $this->upper_bound = $upper_bound;
        $this->noise = new OUProcess($la,
                $mean,$std_dev,
                $theta,$dt,
                $x_initial);
    }

    public function initialize()
    {
        $this->ouProcess->reset();
    }

    /**
    * @param NDArray<any> $states
    * @return NDArray<float> $actions
    */
    public function action(QPolicy $qPolicy, NDArray $states, bool $training) : NDArray
    {
        $la = $this->la;
        $actions = $qPolicy->getQValues($states);

        if($training) {
            $noise = $this->noise->process();
            $actions = $la->add($noise,$la->copy($actions)); // add noise to batch
        }

        $orgShape = $shape = $actions->shape();
        $count = array_shift($shape);
        $size = array_product($shape);
        $actions = $la->transpose($actions->reshape([$count,$size]));

        $flat_lower = $this->lower_bound->reshape([$size]);
        $flat_upper = $this->upper_bound->reshape([$size]);
        for($i=0;$i<$size;$i++) {
            $la->maximum($actions[[$i,$i]],$flat_lower[$i]);
            $la->minimum($actions[[$i,$i]],$flat_upper[$i]);
        }

        $actions = $la->transpose($actions);
        $actions = $actions->reshape($orgShape);
        return $actions;
    }
}
