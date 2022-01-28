<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;

class OUProcess
{
    protected $initialValue;
    protected $damping;
    protected $stddev;
    protected $seed;

    public function __construct(
        $la,
        NDArray $initialValue=null,
        float $damping=null,
        float $stddev=null,
        int $seed=null
    )
    {
        if($damping===null) {
            $damping=0.15;
        }
        if($stddev===null) {
            $stddev=0.2;
        }
        if($initialValue===null) {
            $initialValue=$la->array([0]);
        }
        $this->la = $la;
        $this->damping = min(max($damping,0),1);
        $this->stddev = max($stddev,0);
        $this->seed = $seed;
        $this->initialValue = $initialValue;
        $this->reset();
    }

    public function reset()
    {
        $this->x = $la->copy($initialValue);
    }

    public function __invoke()
    {
        return $this->process();
    }
 
    public function process()
    {
        $la = $this->la;
        $noise = $la->randomNormal(
            $this->x->shape(),
            0,
            $this->stddev,
            $this->x->dtype(),
            $this->seed);
        // x = prev_x * (1-dampling) + N(0,stddev)
        $la->increment($this->x, 0, 1-$this->damping);
        $la->axpy($noise,$this->x);

        // mean = 0 
        // dampling = theta*dt
        // stddev = stddev_*sqrt(dt)
        // x = prev_x + theta*(mean-prev_x)*dt + N(0,stddev_)*sqrt(dt)
        //   = prev_x * (1-theta*dt) + N(0,stddev_)*sqrt(dt)
        //   = prev_x * (1-theta*dt) + N(0,stddev_*sqrt(dt))
        return $x;
    }
}