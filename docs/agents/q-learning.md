# Linear Q-learning

Class: `Rindow\RL\Agents\Agent\QLearning\QLearningAgent`

This agent implements one-step off-policy Q-learning with a sparse `TileCoder`.

```php
new QLearningAgent(
    object $la, TileCoder $tileCoder, int $numActions,
    float $learningRate=0.3, float $gamma=0.99, float $epsilon=0.05,
    ?string $stateField=null, ?string $actionMaskField=null,
    float $initialValue=0.0, ?Builder $nn=null,
)
```

The effective step size per active feature is
`learningRate / activeFeatureCount()`. `initialValue` is divided in the same way
so the sum of active weights equals the requested initial estimate.

`update($state,$action,$reward,$nextState,$terminal)` applies

```text
delta = reward + gamma*max_a Q(nextState,a) - Q(state,action)
w[action,activeFeatures] += alpha*delta
```

The bootstrap term is zero at a terminal state. `update()` returns the TD error.
`value()`, stochastic and deterministic selection, observation parsing, action
masks, and checkpoint save/load are public.

The optional `$nn` is used only to transfer device observations to the host,
because the sparse weight table is maintained as PHP arrays.
