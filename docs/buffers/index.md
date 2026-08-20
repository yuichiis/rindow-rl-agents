# Experience Buffers

- [ReplayBuffer](replay-buffer.md) stores off-policy transitions and samples
  them with replacement.
- [RolloutBuffer](rollout-buffer.md) stores one fixed-size on-policy rollout and
  computes generalized advantage estimates.

Both allocate storage on the supplied linear-algebra backend. Observations and
actions passed to `add()` must therefore be compatible with that backend.
