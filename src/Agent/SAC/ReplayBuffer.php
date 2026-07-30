<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Rindow\NeuralNetworks\Builder\Builder;
use Interop\Polite\Math\Matrix\NDArray;


/**
 * リプレイバッファ
 */
class ReplayBuffer
{
    private object $la;
    private int $capacity;
    private int $obsDim;
    private int $actDim;
    private int $ptr;
    private int $size;
    private NDArray $obs;
    private NDArray $rewards;
    private NDArray $nextObs;
    private NDArray $dones;
    private NDArray $actions;

    public function __construct(
        object $la,
        int $capacity,
        int $obsDim,
        int $actDim
        )
    {
        $this->la = $la;
        $this->capacity = $capacity;
        $this->ptr = 0;
        $this->size = 0;
        $this->obs      = $la->zeros($la->alloc([$capacity, $obsDim], dtype:NDArray::float32));
        $this->rewards  = $la->zeros($la->alloc([$capacity, 1],       dtype:NDArray::float32));
        $this->nextObs = $la->zeros($la->alloc([$capacity, $obsDim], dtype:NDArray::float32));
        $this->dones    = $la->zeros($la->alloc([$capacity, 1],       dtype:NDArray::float32));
        $this->actions  = $la->zeros($la->alloc([$capacity, $actDim], dtype:NDArray::float32));
    }

    public function add(
        NDArray $obs,
        NDArray $action,
        float $reward,
        NDArray $nextObs,
        bool $done
        ) : void
    {
        $this->obs[$this->ptr]      = $obs;
        $this->actions[$this->ptr]  = $action;
        $this->rewards[$this->ptr][0] = $reward;
        $this->nextObs[$this->ptr] = $nextObs;
        $this->dones[$this->ptr][0] = $done;
        $this->ptr  = ($this->ptr + 1) % $this->capacity;
        $this->size = min($this->size + 1, $this->capacity);
    }

    public function sample(int $batchSize) : array
    {
        // PyTorch/NumPy版の np.random.randint と同じく復元抽出にする。
        // randomSequence は非復元抽出なので、SACの更新分布が変わってしまう。
        // randomUniformの整数出力を使う。上限はsize未満で、各要素が独立に
        // 生成されるため、np.random.randintと同じ復元抽出になる。
        $idx = $this->la->randomUniform(
            [$batchSize],
            0,
            $this->size-1,
            dtype:NDArray::int32,
        );
        return [
            $this->la->gather($this->obs,$idx),
            $this->la->gather($this->actions,$idx),
            $this->la->gather($this->rewards,$idx),
            $this->la->gather($this->nextObs,$idx),
            $this->la->gather($this->dones,$idx),
        ];
    }
}
