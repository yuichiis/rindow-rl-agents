# SAC with gSDE

Class: `Rindow\RL\Agents\Agent\SAC\SACGSDEAgent`

This implementation combines Soft Actor-Critic, twin Q networks, automatic
entropy adjustment, and generalized state-dependent exploration.

## Constructor

```php
new SACGSDEAgent(
    Builder $nn, int|array $obsDim, int $actDim, float $actLimit,
    int $gsdeLatentDim, int $hiddenDim,
    float $lrActor, float $lrCritic, float $lrAlpha,
    float $alphaInit, float $gamma, float $tau, int $batchSize,
    ?array $featureLayers=null,
)
```

All optimization parameters are explicit. The target entropy is `-actDim`.

## Action API

- `sampleNoise()` returns a state-independent noise matrix retained over the
  configured interval.
- `selectAction($obs,$noise)` evaluates the state-dependent exploratory action.
- `selectActionDeterministic($obs)` returns the noise-free action.
- Output is scaled and clipped to `[-actLimit,+actLimit]`.

## Update

The target uses the smaller target-Q estimate minus `alpha*logPi`. Separate
gradient tapes update twin critics, actor, and `logAlpha`. Finally the target
critic receives an in-place Polyak update. `update()` returns `critic_loss`,
`actor_loss`, and `alpha`.

`diagnostics()` reports ranges for mean, log standard deviation, log probability,
state-dependent sigma, Q statistics, and gradient RMS values. `alpha()` returns a
`Variable`; `alphaValue()` returns a host float.

Public model properties are `$actor`, `$critic`, and `$criticTarget`. The
checkpoint includes all of them and `logAlpha`.
