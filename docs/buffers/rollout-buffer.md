# RolloutBuffer

Class: `Rindow\RL\Agents\ReplayBuffer\RolloutBuffer`

```php
new RolloutBuffer(
    object $la, int $capacity, int|array $observationDimensions,
    int $actionDimension=1, bool $continuous=false,
    int $actionMaskDimension=0, bool $storePolicyData=false,
)
```

`add()` stores observation, action, reward, termination flag, episode-boundary
flag, value estimate, and optional old log probability and action mask. Adding
past capacity throws `OverflowException`.

`finish($gamma,$gaeLambda,$lastValue=0.0)` walks backward using

```text
delta[t] = reward[t] + gamma*nextValue[t] - value[t]
GAE[t] = delta[t] + gamma*lambda*continuation[t]*GAE[t+1]
return[t] = GAE[t] + value[t]
```

True termination makes `nextValue` zero. An episode boundary stops the recursive
GAE term. These are deliberately separate so truncation may bootstrap without
mixing advantages between episodes.

Without policy data the result is `[obs, actions, advantages, returns]`. With
policy data it is `[obs, actions, oldLogProb, advantages, returns, oldValues]`.
An action-mask tensor is appended when configured. `finish()` clears the buffer.
