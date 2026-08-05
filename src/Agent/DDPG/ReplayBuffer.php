<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Interop\Polite\Math\Matrix\NDArray;

class ReplayBuffer
{
    private object $la;
    private int $capacity;
    private int $ptr = 0;
    private int $size = 0;
    private NDArray $obs;
    private NDArray $actions;
    private NDArray $rewards;
    private NDArray $nextObs;
    private NDArray $dones;

    public function __construct(object $la, int $capacity, int $obsDim, int $actDim)
    {
        if ($capacity < 1) throw new \InvalidArgumentException('capacity must be positive.');
        $this->la = $la;
        $this->capacity = $capacity;
        $this->obs = $la->zeros($la->alloc([$capacity,$obsDim], dtype:NDArray::float32));
        $this->actions = $la->zeros($la->alloc([$capacity,$actDim], dtype:NDArray::float32));
        $this->rewards = $la->zeros($la->alloc([$capacity,1], dtype:NDArray::float32));
        $this->nextObs = $la->zeros($la->alloc([$capacity,$obsDim], dtype:NDArray::float32));
        $this->dones = $la->zeros($la->alloc([$capacity,1], dtype:NDArray::float32));
    }

    public function add(NDArray $obs, NDArray $action, float $reward, NDArray $nextObs, bool $done) : void
    {
        $i = $this->ptr;
        $this->obs[$i] = $obs;
        $this->actions[$i] = $action;
        $this->rewards[$i][0] = $reward;
        $this->nextObs[$i] = $nextObs;
        $this->dones[$i][0] = $done ? 1.0 : 0.0;
        $this->ptr = ($i+1) % $this->capacity;
        $this->size = min($this->size+1, $this->capacity);
    }

    public function size() : int { return $this->size; }

    public function sample(int $batchSize) : array
    {
        if ($this->size === 0) throw new \UnderflowException('Replay buffer is empty.');
        $idx = $this->la->randomUniform([$batchSize], 0, $this->size-1, dtype:NDArray::int32);
        return [$this->la->gather($this->obs,$idx), $this->la->gather($this->actions,$idx),
            $this->la->gather($this->rewards,$idx), $this->la->gather($this->nextObs,$idx),
            $this->la->gather($this->dones,$idx)];
    }
}
