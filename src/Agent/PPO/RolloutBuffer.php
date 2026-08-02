<?php
namespace Rindow\RL\Agents\Agent\PPO;

use Interop\Polite\Math\Matrix\NDArray;

/** PPOのon-policyロールアウトを固定長NDArrayへ直接蓄積する。 */
class RolloutBuffer
{
    private NDArray $observations;
    private NDArray $actions;
    private NDArray $rewards;
    private NDArray $values;
    private NDArray $logProbs;
    private array $terminated = [];
    private array $episodeEnds = [];
    private int $index = 0;
    private bool $continuous;

    public function __construct(
        private object $la,
        private int $capacity,
        int $obsDim,
        int $actionDim = 1,
        bool $continuous = false,
    ) {
        $this->continuous = $continuous;
        $this->observations = $la->zeros($la->alloc([$capacity, $obsDim], dtype:NDArray::float32));
        $this->actions = $continuous
            ? $la->zeros($la->alloc([$capacity, $actionDim], dtype:NDArray::float32))
            : $la->zeros($la->alloc([$capacity], dtype:NDArray::int32));
        $this->rewards = $la->zeros($la->alloc([$capacity], dtype:NDArray::float32));
        $this->values = $la->zeros($la->alloc([$capacity], dtype:NDArray::float32));
        $this->logProbs = $la->zeros($la->alloc([$capacity], dtype:NDArray::float32));
    }

    public function add(
        NDArray $observation,
        mixed $action,
        float $reward,
        bool $terminated,
        bool $episodeEnd,
        float $value,
        float $logProb,
    ) : void {
        if ($this->full()) {
            throw new \OverflowException('PPO rollout buffer is full.');
        }
        $i = $this->index++;
        // Keep data on the backend.  Converting every observation to a PHP
        // array here is particularly expensive with the FFI backends.
        $this->observations[$i] = $observation;
        $this->actions[$i] = $action instanceof NDArray ? $action : $action;
        $this->rewards[$i] = $reward;
        $this->terminated[$i] = $terminated;
        $this->episodeEnds[$i] = $episodeEnd;
        $this->values[$i] = $value;
        $this->logProbs[$i] = $logProb;
    }

    public function full() : bool { return $this->index >= $this->capacity; }
    public function size() : int { return $this->index; }

    /** @return array{NDArray,NDArray,NDArray,NDArray,NDArray,NDArray} */
    public function finish(float $gamma, float $gaeLambda, float $lastValue = 0.0) : array
    {
        $size = $this->index;
        $advantages = $this->la->zeros($this->la->alloc([$size], dtype:NDArray::float32));
        $returns = $this->la->zeros($this->la->alloc([$size], dtype:NDArray::float32));
        $gae = 0.0;
        for ($i = $size - 1; $i >= 0; $i--) {
            $value = (float)$this->values[$i];
            $bootstrap = $this->terminated[$i] ? 0.0 : (
                $i + 1 < $size ? (float)$this->values[$i + 1] : $lastValue
            );
            $continueGae = $this->episodeEnds[$i] ? 0.0 : 1.0;
            $delta = (float)$this->rewards[$i] + $gamma * $bootstrap - $value;
            $gae = $delta + $gamma * $gaeLambda * $continueGae * $gae;
            $advantages[$i] = $gae;
            $returns[$i] = $gae + $value;
        }
        $data = [
            $this->la->slice($this->observations, [0, 0], [$size, $this->observations->shape()[1]]),
            $this->continuous
                ? $this->la->slice($this->actions, [0, 0], [$size, $this->actions->shape()[1]])
                : $this->la->slice($this->actions, [0], [$size]),
            $this->la->slice($this->logProbs, [0], [$size]),
            $advantages,
            $returns,
            $this->la->slice($this->values, [0], [$size]),
        ];
        $this->clear();
        return $data;
    }

    public function clear() : void
    {
        $this->index = 0;
        $this->terminated = [];
        $this->episodeEnds = [];
    }
}
