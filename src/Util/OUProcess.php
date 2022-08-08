<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;

class OUProcess
{
    public function __construct(
        object $la,
        NDArray $mean,
        NDArray $std_deviation,
        float $theta=null,
        float $dt=null,
        NDArray $x_initial=null)
    {
        $theta = $theta ?? 0.15;
        $dt = $dt ?? 1e-2;

        $this->la = $la;
        $this->theta = $theta;
        $this->mean = $mean;
        $this->std_dev = $std_deviation;
        $this->dt = $dt;
        $this->x_initial = $x_initial;
        if($std_deviation!==null) {
            if($mean->shape()!=$std_deviation->shape()) {
                throw new InvalidArgumentException('The shape of mean and std_deviation must be the same');
            }
        }
        if($x_initial!==null) {
            if($mean->shape()!=$x_initial->shape()) {
                throw new InvalidArgumentException('The shape of mean and x_initail must be the same');
            }
        }
        $this->reset();
    }

    public function __invoke()
    {
        return $this->process();
    }
    
    public function process()
    {
        $la = $this->la;
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        // x = x_prev + theta*(mean-x_prev)*dt + std_dev*sqrt(dt)*random(mean.shape)
        $x = $la->axpy(
            $this->x_prev,
            $la->axpy(
                $la->axpy($this->x_prev,$la->copy($this->mean),-1),
                $la->multiply($this->std_dev,$la->randomNormal($this->mean->shape(),0.0, sqrt($this->dt))),
                $this->dt*$this->theta
            )
        );
        # Store x into x_prev
        # Makes next noise dependent on current one
        $this->x_prev = $x;
        return $x;
    }

    public function reset()
    {
        $la = $this->la;
        if($this->x_initial!==null) {
            $this->x_prev = $this->x_initial;
        } else {
            $this->x_prev = $la->zerosLike($this->mean);
        }
    }
}
