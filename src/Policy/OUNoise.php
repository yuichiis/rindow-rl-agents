<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\OUProcess;

class OUNoise extends AbstractPolicy
{
    protected $qPolicy;

    protected $noiseMax = -INF;
    protected $noiseMin = INF;
    protected $actionMax = -INF;
    protected $actionMin = INF;

    public function __construct(
        $la, QPolicy $qPolicy,
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
        $this->qPolicy = $qPolicy;
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
        $this->noiseMax = -INF;
        $this->noiseMin = INF;
        $this->actionMax = -INF;
        $this->actionMin = INF;
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($state,bool $training)
    {
        $la = $this->la;
        $actions = $this->qPolicy->getQValues($state);

        if($training) {
            $noise = $this->noise->process();
            $actions = $la->axpy($noise,$la->copy($actions));
            $this->noiseMax = max($this->noiseMax,$noise[0]);
            $this->noiseMin = min($this->noiseMin,$noise[0]);
        }

        $size = $actions->size();
        $flat_actions = $actions->reshape([$size]);

        $flat_lower = $this->lower_bound->reshape([$size]);
        $flat_upper = $this->upper_bound->reshape([$size]);
        for($i=0;$i<$size;$i++) {
            $la->maximum($flat_actions[[$i,$i]],$flat_lower[$i]);
            $la->minimum($flat_actions[[$i,$i]],$flat_upper[$i]);
        }
        $this->actionMax = max($this->actionMax,$actions[0]);
        $this->actionMin = min($this->actionMin,$actions[0]);
        return $actions;
    }

    public function noiseActionMinMax()
    {
        return [$this->noiseMin,$this->noiseMax,$this->actionMin,$this->actionMax];
    }
}
