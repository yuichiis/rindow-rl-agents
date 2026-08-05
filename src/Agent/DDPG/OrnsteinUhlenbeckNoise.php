<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Interop\Polite\Math\Matrix\NDArray;

/** Temporally correlated exploration process from the original DDPG paper. */
class OrnsteinUhlenbeckNoise
{
    private NDArray $state;

    public function __construct(
        private object $la,
        private int $dimension,
        private float $sigma=0.2,
        private float $theta=0.15,
        private float $dt=0.01,
        private float $mean=0.0,
    ) { $this->reset(); }

    public function reset() : void
    {
        $this->state = $this->la->fill($this->mean, $this->la->alloc([$this->dimension], dtype:NDArray::float32));
    }

    public function sample() : NDArray
    {
        $drift = $this->la->scal($this->theta*$this->dt,
            $this->la->increment($this->la->scal(-1.0, $this->la->copy($this->state)), $this->mean));
        $random = $this->la->randomNormal([$this->dimension], 0.0, 1.0);
        $diffusion = $this->la->scal($this->sigma*sqrt($this->dt), $random);
        $this->state = $this->la->add($this->state, $this->la->add($drift, $diffusion));
        return $this->la->copy($this->state);
    }
}
