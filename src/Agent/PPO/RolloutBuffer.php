<?php
namespace Rindow\RL\Agents\Agent\PPO;

use Interop\Polite\Math\Matrix\NDArray;

/** PPOのon-policyロールアウトとGAEを保持する。 */
class RolloutBuffer
{
    private array $observations = [];
    private array $actions = [];
    private array $rewards = [];
    private array $terminated = [];
    private array $episodeEnds = [];
    private array $values = [];
    private array $nextValues = [];
    private array $logProbs = [];

    public function __construct(private object $la, private int $capacity)
    {
    }

    public function add(
        NDArray $observation,
        int $action,
        float $reward,
        bool $terminated,
        bool $episodeEnd,
        float $value,
        float $nextValue,
        float $logProb,
    ) : void {
        if ($this->full()) {
            throw new \OverflowException('PPO rollout buffer is full.');
        }
        $this->observations[] = $observation->toArray();
        $this->actions[] = $action;
        $this->rewards[] = $reward;
        $this->terminated[] = $terminated;
        $this->episodeEnds[] = $episodeEnd;
        $this->values[] = $value;
        $this->nextValues[] = $nextValue;
        $this->logProbs[] = $logProb;
    }

    public function full() : bool
    {
        return count($this->rewards) >= $this->capacity;
    }

    public function size() : int
    {
        return count($this->rewards);
    }

    /** @return array{NDArray,NDArray,NDArray,NDArray,NDArray,NDArray} */
    public function finish(float $gamma, float $gaeLambda) : array
    {
        $size = $this->size();
        $advantages = array_fill(0, $size, 0.0);
        $returns = array_fill(0, $size, 0.0);
        $gae = 0.0;
        for ($i = $size - 1; $i >= 0; $i--) {
            // terminatedではbootstrapしない。truncatedでは次状態価値を使うが、
            // GAE自体は次の（reset後の）エピソードへ連鎖させない。
            $bootstrap = $this->terminated[$i] ? 0.0 : $this->nextValues[$i];
            $continueGae = $this->episodeEnds[$i] ? 0.0 : 1.0;
            $delta = $this->rewards[$i] + $gamma * $bootstrap - $this->values[$i];
            $gae = $delta + $gamma * $gaeLambda * $continueGae * $gae;
            $advantages[$i] = $gae;
            $returns[$i] = $gae + $this->values[$i];
        }

        $data = [
            $this->la->array($this->observations, dtype:NDArray::float32),
            $this->la->array($this->actions, dtype:NDArray::int32),
            $this->la->array($this->logProbs, dtype:NDArray::float32),
            $this->la->array($advantages, dtype:NDArray::float32),
            $this->la->array($returns, dtype:NDArray::float32),
            $this->la->array($this->values, dtype:NDArray::float32),
        ];
        $this->clear();
        return $data;
    }

    public function clear() : void
    {
        $this->observations = $this->actions = $this->rewards = [];
        $this->terminated = $this->episodeEnds = $this->values = [];
        $this->nextValues = $this->logProbs = [];
    }
}
