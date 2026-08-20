# Off-policy Runners

## DQN

The DQN runner owns a discrete `ReplayBuffer`, epsilon schedule, action masks,
and optional reward/observation transforms.

```php
$runner->train(
    $totalSteps, $learningStarts, $trainEvery, $evalEvery, $evalEpisodes,
    $epsilonStart=1.0, $epsilonEnd=0.05,
    $epsilonDecaySteps=50000, $bestModelFile=null,
);
```

Time-limit truncation keeps a bootstrap value; only true termination is stored
as `done`.

## DDPG

The runner creates a continuous replay buffer and an OU noise process. Before
`startSteps`, actions are uniform random values. Updates begin at `updateAfter`;
grouped updates preserve a one-to-one update/data ratio.

```php
$runner->train($totalSteps,$startSteps,$updateAfter,$updateEvery,
               $evalEvery,$evalEpisodes,$bestModelFile=null);
```

## SAC+gSDE

The SAC runner retains a sampled gSDE noise matrix and resamples every
`gsdeResetFreq` steps. Its training signature is

```php
$runner->train($totalSteps,$startSteps,$updateEvery,$gsdeResetFreq,
               $evalEvery,$evalEpisodes,$evalgSDE=null,$bestModelFile=null);
```

Deterministic evaluation is always recorded; exploratory gSDE evaluation is
optional. Diagnostics and alpha are included in history.

All three runners select best checkpoints primarily by raw evaluation reward,
using transformed reward as a tie-breaker where available.
