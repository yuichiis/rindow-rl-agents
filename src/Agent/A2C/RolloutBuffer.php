<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;

/** Fixed-size, single-environment n-step rollout storage for A2C. */
class RolloutBuffer
{
    private NDArray $observations;
    private NDArray $actions;
    private NDArray $rewards;
    private NDArray $values;
    private ?NDArray $actionMasks = null;
    private array $terminated = [];
    private array $episodeEnds = [];
    private int $index = 0;

    public function __construct(
        private object $la,
        private int $capacity,
        int $obsDim,
        private int $actionDim = 1,
        private bool $continuous = false,
        int $maskDim = 0,
    )
    {
        if ($capacity < 1) {
            throw new \InvalidArgumentException('capacity must be greater than zero.');
        }
        $this->observations = $la->zeros($la->alloc([$capacity, $obsDim], dtype:NDArray::float32));
        $this->actions = $continuous
            ? $la->zeros($la->alloc([$capacity, $actionDim], dtype:NDArray::float32))
            : $la->zeros($la->alloc([$capacity], dtype:NDArray::int32));
        $this->rewards = $la->zeros($la->alloc([$capacity], dtype:NDArray::float32));
        $this->values = $la->zeros($la->alloc([$capacity], dtype:NDArray::float32));
        if ($maskDim > 0) {
            $this->actionMasks = $la->zeros(
                $la->alloc([$capacity, $maskDim], dtype:NDArray::bool)
            );
        }
    }

    public function add(NDArray $observation, int|NDArray $action, float $reward, bool $terminated,
        bool $episodeEnd, float $value, ?NDArray $actionMask = null) : void
    {
        if ($this->full()) {
            throw new \OverflowException('A2C rollout buffer is full.');
        }
        $i = $this->index++;
        $this->observations[$i] = $observation;
        $this->actions[$i] = $action;
        $this->rewards[$i] = $reward;
        $this->values[$i] = $value;
        $this->terminated[$i] = $terminated;
        $this->episodeEnds[$i] = $episodeEnd;
        if ($this->actionMasks !== null) {
            if ($actionMask === null) {
                throw new \InvalidArgumentException('Action mask is required for this rollout buffer.');
            }
            $this->actionMasks[$i] = $actionMask;
        } elseif ($actionMask !== null) {
            throw new \InvalidArgumentException('This rollout buffer is not configured for action masks.');
        }
    }

    public function full() : bool { return $this->index >= $this->capacity; }

    /** @return array{NDArray,NDArray,NDArray,NDArray} observations, actions, advantages, returns */
    public function finish(float $gamma, float $gaeLambda, float $lastValue = 0.0) : array
    {
        $size = $this->index;
        if ($size === 0) {
            throw new \UnderflowException('A2C rollout buffer is empty.');
        }
        $advantages = $this->la->zeros($this->la->alloc([$size], dtype:NDArray::float32));
        $returns = $this->la->zeros($this->la->alloc([$size], dtype:NDArray::float32));
        $gae = 0.0;
        for ($i = $size - 1; $i >= 0; $i--) {
            $value = (float)$this->values[$i];
            $nextValue = $this->terminated[$i] ? 0.0
                : ($i + 1 < $size ? (float)$this->values[$i + 1] : $lastValue);
            $delta = (float)$this->rewards[$i] + $gamma * $nextValue - $value;
            $gae = $delta + $gamma * $gaeLambda * ($this->episodeEnds[$i] ? 0.0 : 1.0) * $gae;
            $advantages[$i] = $gae;
            $returns[$i] = $gae + $value;
        }
        $rollout = [
            $this->la->slice($this->observations, [0, 0], [$size, $this->observations->shape()[1]]),
            $this->continuous
                ? $this->la->slice($this->actions, [0, 0], [$size, $this->actionDim])
                : $this->la->slice($this->actions, [0], [$size]),
            $advantages,
            $returns,
        ];
        if ($this->actionMasks !== null) {
            $rollout[] = $this->la->slice(
                $this->actionMasks, [0, 0], [$size, $this->actionMasks->shape()[1]]
            );
        }
        $this->index = 0;
        $this->terminated = [];
        $this->episodeEnds = [];
        return $rollout;
    }
}
