<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Interop\Polite\Math\Matrix\NDArray;

class ReplayBuffer
{
    private int $ptr = 0;
    private int $size = 0;
    private NDArray $observations;
    private NDArray $actions;
    private NDArray $rewards;
    private NDArray $nextObservations;
    private NDArray $dones;

    public function __construct(
        private object $la,
        private int $capacity,
        int $obsDim,
    ) {
        if ($capacity < 1 || $obsDim < 1) {
            throw new \InvalidArgumentException('Capacity and observation dimension must be positive.');
        }
        $this->observations = $la->zeros($la->alloc([$capacity,$obsDim], dtype:NDArray::float32));
        $this->actions = $la->zeros($la->alloc([$capacity], dtype:NDArray::int32));
        $this->rewards = $la->zeros($la->alloc([$capacity], dtype:NDArray::float32));
        $this->nextObservations = $la->zeros($la->alloc([$capacity,$obsDim], dtype:NDArray::float32));
        $this->dones = $la->zeros($la->alloc([$capacity], dtype:NDArray::float32));
    }

    public function add(
        NDArray $observation,
        int $action,
        float $reward,
        NDArray $nextObservation,
        bool $done,
    ) : void {
        $i = $this->ptr;
        $this->observations[$i] = $observation;
        $this->actions[$i] = $action;
        $this->rewards[$i] = $reward;
        $this->nextObservations[$i] = $nextObservation;
        $this->dones[$i] = $done ? 1.0 : 0.0;
        $this->ptr = ($i+1) % $this->capacity;
        $this->size = min($this->size+1, $this->capacity);
    }

    public function size() : int { return $this->size; }

    /** @return array{NDArray,NDArray,NDArray,NDArray,NDArray} */
    public function sample(int $batchSize) : array
    {
        if ($batchSize < 1) throw new \InvalidArgumentException('batchSize must be positive.');
        if ($this->size === 0) throw new \UnderflowException('Replay buffer is empty.');
        $indices = $this->la->randomUniform([$batchSize], 0, $this->size-1, dtype:NDArray::int32);
        return [
            $this->la->gather($this->observations,$indices),
            $this->la->gather($this->actions,$indices),
            $this->la->gather($this->rewards,$indices),
            $this->la->gather($this->nextObservations,$indices),
            $this->la->gather($this->dones,$indices),
        ];
    }
}
