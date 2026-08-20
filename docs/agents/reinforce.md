# REINFORCE

Class: `Rindow\RL\Agents\Agent\Reinforce\ReinforceAgent`

REINFORCE learns a categorical policy from complete Monte Carlo episodes.

```php
new ReinforceAgent(
    Builder $nn, int $obsDim, int $numActions,
    array $hiddenLayers=[128], float $learningRate=1e-2,
    float $entropyWeight=0.0, float $maxGradNorm=1.0,
    string $activation='relu',
)
```

`selectAction()` samples the softmax policy. `selectActionDeterministic()` returns
the largest-probability action. `update($observations,$actions,$returns)` requires
equal non-zero leading dimensions and minimizes

```text
-mean(log pi(action|state) * return) - entropyWeight*entropy
```

It returns `policy_loss` and `entropy`. The runner computes discounted returns
and can normalize them within each episode. The public `$network` property gives
access to the policy network. Checkpoints validate observation and action sizes.
