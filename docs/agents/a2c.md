# A2C

Class: `Rindow\RL\Agents\Agent\A2C\A2CAgent`

A2C uses one actor-critic model and synchronous on-policy rollouts. It supports
categorical discrete policies and diagonal Gaussian continuous policies.

## Constructor

```php
new A2CAgent(
    Builder $nn, int|array $obsDim, int $numActions,
    array $hiddenLayers=[64,64], float $learningRate=7e-4,
    float $valueLossWeight=0.5, float $entropyWeight=0.01,
    float $maxGradNorm=0.5, bool $normalizeAdvantages=false,
    bool $continuous=false, ?NDArray $actionMin=null,
    ?NDArray $actionMax=null, float $initialLogStd=-0.5,
    string $optimizer='adam', mixed $actionKernelInitializer=null,
    string $activation='tanh', ?string $stateField=null,
    ?string $actionMaskField=null, ?array $featureLayers=null,
)
```

Continuous actions require `actionMin` and `actionMax`; action masks are only
valid for discrete policies. `optimizer` accepts `adam` or `rmsprop`.

## Methods

- `selectAction()` returns `[action, value]`.
- `selectActionDeterministic()` returns argmax for a categorical policy or the
  clipped mean for a Gaussian policy.
- `value()` returns the scalar V(s).
- `clipAction()` applies continuous action bounds.
- `update($rollout)` returns `policy_loss`, `value_loss`, `entropy`, and `std`.

Discrete policy loss is the negative mean selected log probability weighted by
the advantage. Continuous loss uses a diagonal Gaussian likelihood. The total
loss combines policy loss, weighted value error, and entropy bonus.

## Rollout format

`[observations, actions, advantages, returns]`, optionally followed by action
masks. The A2C runner produces this format with `RolloutBuffer::finish()`.

The public `$network` property exposes the underlying actor-critic model.
