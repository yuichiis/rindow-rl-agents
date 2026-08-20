# PPO

Class: `Rindow\RL\Agents\Agent\PPO\PPOAgent`

PPO performs multiple minibatch epochs over an on-policy rollout while clipping
the policy probability ratio. It supports categorical, diagonal Gaussian, and
generalized state-dependent exploration policies.

## Constructor

```php
new PPOAgent(
    Builder $nn, int|array $obsDim, int $numActions,
    array $hiddenLayers=[64,64], float $learningRate=3e-4,
    float $clipRange=0.2, float $valueLossWeight=0.5,
    float $entropyWeight=0.01, int $epochs=10, int $batchSize=64,
    float $maxGradNorm=0.5, bool $clipValueLoss=true,
    bool $sharedBackbone=false, bool $continuous=false,
    ?NDArray $actionMin=null, ?NDArray $actionMax=null,
    string $exploration='gaussian', int $sdeSampleFreq=-1,
    float $sdeInitialLogStd=-2.0, ?string $stateField=null,
    ?string $actionMaskField=null, ?array $featureLayers=null,
)
```

`exploration` is `gaussian` or `gsde`. gSDE requires `continuous:true` and
`sharedBackbone:true`. `featureLayers` also requires a shared backbone.

## Rollout format

```text
[observations, actions, oldLogProbabilities,
 advantages, returns, oldValues, optionalActionMasks]
```

Advantages are normalized before minibatch updates. `update()` returns averaged
`policy_loss`, `value_loss`, and `entropy`.

## Objective

```text
ratio = exp(newLogProb - oldLogProb)
policy loss = -mean(min(ratio*A, clip(ratio,1-e,1+e)*A))
```

Value clipping optionally limits the value change relative to `oldValues`.
Global norm clipping is applied before each optimizer update.

## Selection and gSDE

`selectAction()` returns `[action, logProbability, value]`.
`selectActionDeterministic()` uses argmax or the clipped mean. Call
`resetExplorationNoise()` according to `sdeSampleFreq()` when managing gSDE
without the bundled runner.
