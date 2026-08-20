# On-policy Runners

## A2C

Constructor parameters include training/evaluation environments, agent,
`rolloutSteps=5`, `gamma=0.99`, `gaeLambda=1.0`, optional reward and observation
callbacks, truncation bootstrapping, and solved criteria.

```php
$history = $runner->train($totalSteps, $evalEvery=5000,
                          $evalEpisodes=10, $bestModelFile=null);
```

History includes step, training reward/length, evaluation reward, policy/value
loss, entropy, and continuous-policy standard deviation.

## PPO

The PPO runner defaults to `rolloutSteps=2048`, `gamma=0.99`, and
`gaeLambda=0.95`. It stores old log probabilities and values required by PPO.
`train()` has the same four arguments as A2C and returns raw/transformed
evaluation metrics plus averaged PPO losses.

## REINFORCE

The episode runner accepts `gamma=0.99` and `normalizeReturns=true`. It collects a
complete episode, computes discounted returns backward, optionally standardizes
them, then performs one policy update.

```php
$history = $runner->train($totalEpisodes, $evalEvery=50,
                          $evalEpisodes=10, $bestModelFile=null);
```
