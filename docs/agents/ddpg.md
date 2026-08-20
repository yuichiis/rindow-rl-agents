# DDPG

Class: `Rindow\RL\Agents\Agent\DDPG\DDPGAgent`

DDPG is an off-policy deterministic actor-critic for bounded continuous actions.

## Constructor

```php
new DDPGAgent(
    Builder $nn, int|array $obsDim, int $actDim, float $actLimit,
    int $hiddenDim=256, float $lrActor=1e-4, float $lrCritic=1e-3,
    float $gamma=0.99, float $tau=0.005, int $batchSize=128,
    ?array $featureLayers=null,
)
```

Actor output passes through `tanh`, is multiplied by `actLimit`, and is clipped
to `[-actLimit,+actLimit]`. `selectAction($obs,$noise)` adds noise in environment
action units before the final clipping. `selectActionDeterministic()` omits noise.

## Update

The critic fits `r + gamma*(1-done)*Qtarget(s', actorTarget(s'))`. The actor
minimizes the negative critic value of its actions. Actor and critic targets are
then updated in place with

```text
target = tau*online + (1-tau)*target
```

`update()` returns `actor_loss` and `critic_loss`.

Public model properties are `$actor`, `$actorTarget`, `$critic`, and
`$criticTarget`. Checkpoints contain all four models.

`OrnsteinUhlenbeckNoise` provides temporally correlated exploration with
`sample()` and `reset()`; the DDPG runner resets it at episode boundaries.
