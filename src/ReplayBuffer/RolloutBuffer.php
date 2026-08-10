<?php
namespace Rindow\RL\Agents\ReplayBuffer;

use Interop\Polite\Math\Matrix\NDArray;

/** Fixed-size, single-environment on-policy rollout storage. */
class RolloutBuffer
{
    private NDArray $observations;
    private NDArray $actions;
    private NDArray $rewards;
    private NDArray $values;
    private ?NDArray $logProbabilities = null;
    private ?NDArray $actionMasks = null;
    private array $terminated = [];
    private array $episodeEnds = [];
    private int $index = 0;

    public function __construct(
        private object $la,
        private int $capacity,
        int|array $observationDimensions,
        private int $actionDimension=1,
        private bool $continuous=false,
        int $actionMaskDimension=0,
        private bool $storePolicyData=false,
    ) {
        $observationShape = is_int($observationDimensions)
            ? [$observationDimensions]
            : array_values($observationDimensions);
        if ($capacity < 1 || $observationShape === []
            || array_filter($observationShape,static fn($dim)=>!is_int($dim) || $dim < 1)) {
            throw new \InvalidArgumentException(
                'Capacity and observation dimensions must be positive.'
            );
        }
        if ($actionDimension < 1) {
            throw new \InvalidArgumentException('Action dimension must be positive.');
        }
        if ($actionMaskDimension < 0) {
            throw new \InvalidArgumentException('Action mask dimension must not be negative.');
        }
        if ($continuous && $actionMaskDimension !== 0) {
            throw new \InvalidArgumentException(
                'Action masks are only supported for discrete actions.'
            );
        }
        $this->observations = $la->zeros($la->alloc(
            array_merge([$capacity],$observationShape),dtype:NDArray::float32
        ));
        $this->actions = $continuous
            ? $la->zeros($la->alloc([$capacity,$actionDimension],dtype:NDArray::float32))
            : $la->zeros($la->alloc([$capacity],dtype:NDArray::int32));
        $this->rewards = $la->zeros($la->alloc([$capacity],dtype:NDArray::float32));
        $this->values = $la->zeros($la->alloc([$capacity],dtype:NDArray::float32));
        if ($storePolicyData) {
            $this->logProbabilities = $la->zeros(
                $la->alloc([$capacity],dtype:NDArray::float32)
            );
        }
        if ($actionMaskDimension > 0) {
            $this->actionMasks = $la->zeros(
                $la->alloc([$capacity,$actionMaskDimension],dtype:NDArray::bool)
            );
        }
    }

    public function add(
        NDArray $observation,
        int|NDArray $action,
        float $reward,
        bool $terminated,
        bool $episodeEnd,
        float|NDArray $value,
        float|NDArray|null $logProbability=null,
        ?NDArray $actionMask=null,
    ) : void {
        if ($this->full()) {
            throw new \OverflowException('Rollout buffer is full.');
        }
        if ($this->storePolicyData && $logProbability === null) {
            throw new \InvalidArgumentException('A log probability is required.');
        }
        if (!$this->storePolicyData && $logProbability !== null) {
            throw new \InvalidArgumentException(
                'This rollout buffer does not store policy data.'
            );
        }
        if ($this->actionMasks !== null && $actionMask === null) {
            throw new \InvalidArgumentException('An action mask is required.');
        }
        if ($this->actionMasks === null && $actionMask !== null) {
            throw new \InvalidArgumentException(
                'This rollout buffer is not configured for action masks.'
            );
        }

        $i = $this->index++;
        $this->observations[$i] = $observation;
        $actionValue = $action;
        $rewardValue = $reward;
        $stateValue = $value;
        $logProbabilityValue = $logProbability;
        if (!($action instanceof NDArray)) {
            $actionValue = $this->la->array($action,dtype:NDArray::int32);
        }
        $rewardValue = $this->la->array($reward,dtype:NDArray::float32);
        if (!($value instanceof NDArray)) {
            $stateValue = $this->la->array($value,dtype:NDArray::float32);
        }
        if ($logProbability !== null && !($logProbability instanceof NDArray)) {
            $logProbabilityValue = $this->la->array(
                $logProbability,dtype:NDArray::float32
            );
        }
        $this->actions[$i] = $actionValue;
        $this->rewards[$i] = $rewardValue;
        $this->values[$i] = $stateValue;
        $this->terminated[$i] = $terminated;
        $this->episodeEnds[$i] = $episodeEnd;
        if ($this->logProbabilities !== null) {
            $this->logProbabilities[$i] = $logProbabilityValue;
        }
        if ($this->actionMasks !== null) {
            $this->actionMasks[$i] = $actionMask;
        }
    }

    public function full() : bool
    {
        return $this->index >= $this->capacity;
    }

    public function size() : int
    {
        return $this->index;
    }

    public function finish(float $gamma, float $gaeLambda, float $lastValue=0.0) : array
    {
        $size = $this->index;
        if ($size === 0) {
            throw new \UnderflowException('Rollout buffer is empty.');
        }
        $advantageValues = array_fill(0,$size,0.0);
        $returnValues = array_fill(0,$size,0.0);
        $rewardValues = $this->la->toNDArray(
            $this->la->slice($this->rewards,[0],[$size])
        )->toArray();
        $stateValues = $this->la->toNDArray(
            $this->la->slice($this->values,[0],[$size])
        )->toArray();
        $gae = 0.0;
        for ($i = $size-1; $i >= 0; $i--) {
            $value = $stateValues[$i];
            $nextValue = $this->terminated[$i]
                ? 0.0
                : ($i+1 < $size ? $stateValues[$i+1] : $lastValue);
            $delta = $rewardValues[$i]+$gamma*$nextValue-$value;
            $gae = $delta+$gamma*$gaeLambda*($this->episodeEnds[$i] ? 0.0 : 1.0)*$gae;
            $advantageValues[$i] = $gae;
            $returnValues[$i] = $gae+$value;
        }
        $advantages = $this->la->array($advantageValues,dtype:NDArray::float32);
        $returns = $this->la->array($returnValues,dtype:NDArray::float32);

        $indices = $this->la->array(range(0,$size-1),dtype:NDArray::int32);
        $observations = $this->la->gather($this->observations,$indices);
        $actions = $this->continuous
            ? $this->la->slice(
                $this->actions,[0,0],[$size,$this->actionDimension]
            )
            : $this->la->slice($this->actions,[0],[$size]);

        if ($this->storePolicyData) {
            $rollout = [
                $observations,
                $actions,
                $this->la->slice($this->logProbabilities,[0],[$size]),
                $advantages,
                $returns,
                $this->la->slice($this->values,[0],[$size]),
            ];
        } else {
            $rollout = [$observations,$actions,$advantages,$returns];
        }
        if ($this->actionMasks !== null) {
            $rollout[] = $this->la->slice(
                $this->actionMasks,[0,0],[$size,$this->actionMasks->shape()[1]]
            );
        }
        $this->clear();
        return $rollout;
    }

    public function clear() : void
    {
        $this->index = 0;
        $this->terminated = [];
        $this->episodeEnds = [];
    }
}
