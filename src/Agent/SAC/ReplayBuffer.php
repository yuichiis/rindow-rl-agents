<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Rindow\NeuralNetworks\Builder\Builder;
use Interop\Polite\Math\Matrix\NDArray;


# ─────────────────────────────────────────────
# リプレイバッファ  (numpy のまま、変更なし)
# ─────────────────────────────────────────────
class ReplayBuffer
{
    private object $la;
    private int $capacity;
    private int $obs_dim;
    private int $act_dim;
    private int $ptr;
    private int $size;
    private NDArray $obs;
    private NDArray $rewards;
    private NDArray $next_obs;
    private NDArray $dones;
    private NDArray $actions;

    public function __construct(
        object $la,
        int $capacity,
        int $obs_dim,
        int $act_dim
        )
    {
        $this->la = $la;
        $this->capacity = $capacity;
        $this->ptr = 0;
        $this->size = 0;
        $this->obs      = $la->zeros($la->alloc([$capacity, $obs_dim], dtype:NDArray::float32));
        $this->rewards  = $la->zeros($la->alloc([$capacity, 1],       dtype:NDArray::float32));
        $this->next_obs = $la->zeros($la->alloc([$capacity, $obs_dim], dtype:NDArray::float32));
        $this->dones    = $la->zeros($la->alloc([$capacity, 1],       dtype:NDArray::float32));
        $this->actions  = $la->zeros($la->alloc([$capacity, $act_dim], dtype:NDArray::float32));
    }

    public function add(
        NDArray $obs,
        NDArray $action,
        float $reward,
        NDArray $next_obs,
        bool $done
        ) : void
    {
        $this->obs[$this->ptr]      = $obs;
        $this->actions[$this->ptr]  = $action;
        $this->rewards[$this->ptr][0] = $reward;
        $this->next_obs[$this->ptr] = $next_obs;
        $this->dones[$this->ptr][0] = $done;
        $this->ptr  = ($this->ptr + 1) % $this->capacity;
        $this->size = min($this->size + 1, $this->capacity);
    }

    public function sample(int $batch_size) : array
    {
        // PyTorch/NumPy版の np.random.randint と同じく復元抽出にする。
        // randomSequence は非復元抽出なので、SACの更新分布が変わってしまう。
        // randomUniformの整数出力を使う。上限はsize未満で、各要素が独立に
        // 生成されるため、np.random.randintと同じ復元抽出になる。
        $idx = $this->la->randomUniform(
            [$batch_size],
            0,
            $this->size-1,
            dtype:NDArray::int32,
        );
        return [
            $this->la->gather($this->obs,$idx),
            $this->la->gather($this->actions,$idx),
            $this->la->gather($this->rewards,$idx),
            $this->la->gather($this->next_obs,$idx),
            $this->la->gather($this->dones,$idx),
        ];
    }
}
