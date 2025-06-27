<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Util\OUProcess;
use function Rindow\Math\Matrix\R;

class OUNoise extends AbstractPolicy
{
    protected $estimator;
    protected $lower_bound;
    protected $upper_bound;
    protected $noise;

    public function __construct(
        object $la,
        NDArray $mean,
        NDArray $std_dev,
        NDArray $lower_bound,
        NDArray $upper_bound,
        ?float $theta=null,
        ?float $dt=null,
        ?NDArray $x_initial=null
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

    public function isContinuousActions() : bool
    {
        return true;
    }

    public function initialize() : void
    {
        $this->ouProcess->reset();
    }

    /**
    * param  NDArray $states  : (batches,...StateDims)  typeof int32 or float32
    * return NDArray $actions : (batches,...ActionDims) typeof float32
    */
    public function actions(Estimator $estimator, NDArray $states, bool $training, ?NDArray $masks) : NDArray
    {
        $la = $this->la;
        $actions = $estimator->getActionValues($states);

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
            $la->maximum($actions[R($i,$i+1)],$flat_lower[$i]);
            $la->minimum($actions[R($i,$i+1)],$flat_upper[$i]);
        }

        $actions = $la->transpose($actions);
        $actions = $actions->reshape($orgShape);
        return $actions;
    }
}
