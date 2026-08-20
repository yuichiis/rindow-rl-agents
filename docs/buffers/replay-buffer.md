# ReplayBuffer

Class: `Rindow\RL\Agents\ReplayBuffer\ReplayBuffer`

```php
new ReplayBuffer(
    object $la, int $capacity, int|array $observationDimensions,
    ?int $actionDimension=null, int $actionMaskDimension=0,
)
```

When `actionDimension` is `null`, actions are discrete `int32` scalars and
rewards/dones have shape `[capacity]`. Otherwise actions are float32 vectors and
rewards/dones have shape `[capacity,1]`. Observations are float32.

`add($observation,$action,$reward,$nextObservation,$done,$nextActionMask=null)`
overwrites the oldest transition after capacity is reached. A configured mask is
required for every discrete transition. Continuous buffers do not support masks.

`sample($batchSize)` samples indices uniformly with replacement and returns:

```text
continuous: [observations, actions, rewards, nextObservations, dones]
discrete:   [observations, actions, rewards, nextObservations, dones,
             nextActionMasks|null]
```

`sample()` throws `UnderflowException` when empty. `size()` never exceeds
capacity.
