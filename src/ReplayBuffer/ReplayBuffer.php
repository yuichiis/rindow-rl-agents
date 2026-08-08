<?php
namespace Rindow\RL\Agents\ReplayBuffer;

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
    private ?NDArray $nextActionMasks = null;
    private bool $continuousActions;
    private bool $accelerated;

    public function __construct(
        private object $la,
        private int $capacity,
        int|array $observationDimensions,
        ?int $actionDimension=null,
        int $actionMaskDimension=0,
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
        if ($actionDimension !== null && $actionDimension < 1) {
            throw new \InvalidArgumentException('Action dimension must be positive.');
        }
        if ($actionMaskDimension < 0) {
            throw new \InvalidArgumentException('Action mask dimension must not be negative.');
        }
        if ($actionDimension !== null && $actionMaskDimension !== 0) {
            throw new \InvalidArgumentException(
                'Action masks are only supported for discrete actions.'
            );
        }

        $bufferShape = array_merge([$capacity],$observationShape);
        $continuousActions = $actionDimension !== null;
        $this->continuousActions = $continuousActions;
        $this->accelerated = $la->accelerated();
        $this->observations = $la->zeros($la->alloc($bufferShape,dtype:NDArray::float32));
        $this->actions = $la->zeros($la->alloc(
            $continuousActions ? [$capacity,$actionDimension] : [$capacity],
            dtype:$continuousActions ? NDArray::float32 : NDArray::int32,
        ));
        $scalarShape = $continuousActions ? [$capacity,1] : [$capacity];
        $this->rewards = $la->zeros($la->alloc($scalarShape,dtype:NDArray::float32));
        $this->nextObservations = $la->zeros($la->alloc($bufferShape,dtype:NDArray::float32));
        $this->dones = $la->zeros($la->alloc($scalarShape,dtype:NDArray::float32));
        if ($actionMaskDimension > 0) {
            $this->nextActionMasks = $la->zeros(
                $la->alloc([$capacity,$actionMaskDimension],dtype:NDArray::bool)
            );
        }
    }

    public function add(
        NDArray $observation,
        int|NDArray $action,
        float $reward,
        NDArray $nextObservation,
        bool $done,
        ?NDArray $nextActionMask=null,
    ) : void {
        $i = $this->ptr;
        $this->observations[$i] = $observation;
        $actionValue = $action;
        $rewardValue = $reward;
        $doneValue = $done ? 1.0 : 0.0;
        if ($this->accelerated) {
            $actionValue = $action instanceof NDArray ? $action : $this->la->array(
                $action,
                dtype:$this->continuousActions ? NDArray::float32 : NDArray::int32,
            );
            $rewardValue = $this->la->array($reward,dtype:NDArray::float32);
            $doneValue = $this->la->array($doneValue,dtype:NDArray::float32);
        }
        $this->actions[$i] = $actionValue;
        if ($this->continuousActions) {
            $this->rewards[$i][0] = $rewardValue;
            $this->dones[$i][0] = $doneValue;
        } else {
            $this->rewards[$i] = $rewardValue;
            $this->dones[$i] = $doneValue;
        }
        $this->nextObservations[$i] = $nextObservation;
        if ($this->nextActionMasks !== null) {
            if ($nextActionMask === null) {
                throw new \InvalidArgumentException('A next action mask is required.');
            }
            $this->nextActionMasks[$i] = $nextActionMask;
        } elseif ($nextActionMask !== null) {
            throw new \InvalidArgumentException('This replay buffer does not use action masks.');
        }
        $this->ptr = ($i+1) % $this->capacity;
        $this->size = min($this->size+1,$this->capacity);
    }

    public function size() : int
    {
        return $this->size;
    }

    public function sample(int $batchSize) : array
    {
        if ($batchSize < 1) {
            throw new \InvalidArgumentException('Batch size must be positive.');
        }
        if ($this->size === 0) {
            throw new \UnderflowException('Replay buffer is empty.');
        }
        // Sampling with replacement, equivalent to NumPy's randint.
        $indices = $this->la->randomUniform(
            [$batchSize],0,$this->size-1,dtype:NDArray::int32
        );
        $batch = [
            $this->la->gather($this->observations,$indices),
            $this->la->gather($this->actions,$indices),
            $this->la->gather($this->rewards,$indices),
            $this->la->gather($this->nextObservations,$indices),
            $this->la->gather($this->dones,$indices),
        ];
        if (!$this->continuousActions) {
            $batch[] = $this->nextActionMasks === null
                ? null
                : $this->la->gather($this->nextActionMasks,$indices);
        }
        return $batch;
    }
}
