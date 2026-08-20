# Evaluation

Training and evaluation environments are separate constructor arguments.
Evaluation uses deterministic policy methods and does not add epsilon, Gaussian,
or OU exploration. SAC may optionally evaluate with gSDE noise.

`evaluate()` returns mean raw reward. Runners exposing `evaluateDetailed()`
return:

```php
[
    'rawReward' => float,
    'transformedReward' => float,
    'steps' => float,
]
```

Raw reward determines task success. Transformed reward describes the training
signal and can distinguish checkpoints tied on raw score. An episode ends on
either termination or truncation.

When `bestModelFile` is set, the runner saves only improvements. `solvedReward`
and `solvedEvaluations` control early completion independently of checkpoint
saving.
