# True Online Sarsa(λ)

Classes:

- `Rindow\RL\Agents\Agent\Sarsa\TrueOnlineSarsaLambdaAgent`
- `Rindow\RL\Agents\Agent\Sarsa\TileCoder`

## Agent constructor

```php
new TrueOnlineSarsaLambdaAgent(
    object $la, TileCoder $tileCoder, int $numActions,
    float $learningRate=0.3, float $gamma=1.0, float $lambda=0.9,
    float $epsilon=0.0, ?string $stateField=null,
    ?string $actionMaskField=null, float $initialValue=0.0,
    ?Builder $nn=null,
)
```

Call `startEpisode()` before each training episode. It clears Dutch eligibility
traces and resets the previous Q value. For a non-terminal update, `nextAction`
is required because Sarsa evaluates the action selected by the behavior policy.
`update()` returns the TD error.

## TileCoder

```php
new TileCoder(array $low, array $high, int $numTilings=8,
              int $tilesPerDimension=8)
```

`encode()` returns one active integer feature per tiling. Values outside the
configured finite bounds are clipped to edge tiles. `featureCount()`,
`activeFeatureCount()`, and `observationDimension()` describe the encoding.

The agent supports dictionary observations and action masks in the same form as
linear Q-learning. Checkpoints contain weights but not active episode traces.
