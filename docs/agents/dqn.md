# DQN and Double DQN

Class: `Rindow\RL\Agents\Agent\DQN\DQNAgent`

DQN learns a discrete state-action value function. Setting `ddqn:true` makes the
online network select the next action while the target network evaluates it.

## Constructor

```php
new DQNAgent(
    Builder $nn, int|array $obsDim, int $numActions,
    array $hiddenLayers=[128,128], float $learningRate=1e-3,
    float $gamma=0.99, int $batchSize=64,
    int $targetUpdateInterval=500, float $maxGradNorm=10.0,
    ?string $stateField=null, ?string $actionMaskField=null,
    ?array $featureLayers=null, bool $ddqn=false,
)
```

`numActions` must be at least two. `featureLayers` are cloned separately for the
online and target networks. A dictionary observation requires `stateField`; an
action mask additionally requires `actionMaskField`.

## Action selection

- `selectAction($observation, $epsilon=0.0): int` uses epsilon-greedy selection.
- `selectActionDeterministic($observation): int` returns the enabled action with
  the largest Q value.
- `parseObservation()` returns `[state, mask]` after validation.

Disabled Q values are filled with `-1e9` before device-side `reduceArgMax` or
`reduceMax`. The mask must have shape `[numActions]` and allow at least one action.

## Update

`update(ReplayBuffer $buffer)` minimizes

```text
mean((Q(s,a) - [r + gamma*(1-done)*Qtarget(s',a')])^2)
```

It returns `['loss'=>float, 'q_value'=>float]`. Every
`targetUpdateInterval` updates, all target weights are copied in place from the
online network.

## Public state and persistence

`$qNetwork` and `$targetNetwork` are public. `summary()`, dimension accessors,
`saveWeightsToFile()`, and `loadWeightsFromFile()` are provided. A checkpoint
contains both networks and validates `obsDim` and `numActions` when loaded.

See [ReplayBuffer](../buffers/replay-buffer.md) and
[off-policy runners](../runners/off-policy-runners.md).
