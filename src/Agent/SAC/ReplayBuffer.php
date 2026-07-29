<?php
namespace Rindow\RL\Agents\Agent\SAC;

# ─────────────────────────────────────────────
# リプレイバッファ  (numpy のまま、変更なし)
# ─────────────────────────────────────────────
class ReplayBuffer
{
    private Builder $nn;
    private object $la;
    private object $g;
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
        Builder $nn,
        int $capacity,
        int $obs_dim,
        int $act_dim
        )
    {
        $la = $nn->backend()->primaryLA();
        $this->nn = $nn;
        $this->la = $la;
        $this->g = $this->nn->gradient();
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
        $idx = $this->la->randomSequence($this->size, $batch_size);
        return [
            $this->la->gather($this->obs,$idx),
            $this->la->gather($this->actions,$idx),
            $this->la->gather($this->rewards,$idx),
            $this->la->gather($this->next_obs,$idx),
            $this->la->gather($this->dones,$idx),
        ];
    }
}
