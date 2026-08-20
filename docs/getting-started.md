# Getting Started

## Create the numerical backend

```php
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->backend()->primaryLA();
```

The backend selected by Rindow Neural Networks determines where tensors and
model operations run. See [CPU and GPU backends](backends/cpu-and-gpu.md).

## Construct an agent

```php
use Rindow\RL\Agents\Agent\DQN\DQNAgent;

$agent = new DQNAgent(
    $nn,
    obsDim: 4,
    numActions: 2,
    hiddenLayers: [128, 128],
);
```

## Use a runner

Runners own the interaction loop, storage, evaluation schedule, and optional
best-model checkpoint. Training and evaluation environments should be separate
instances so evaluation resets do not disturb the training trajectory.

```php
use Rindow\RL\Agents\Agent\DQN\Runner;

$runner = new Runner(
    $la, $trainingEnv, $evaluationEnv, $agent,
    obsDim: 4,
    bufferSize: 100_000,
);

$history = $runner->train(
    totalSteps: 100_000,
    learningStarts: 1_000,
    trainEvery: 1,
    epsilonStart: 1.0,
    epsilonEnd: 0.05,
    epsilonDecaySteps: 50_000,
    evalEvery: 5_000,
    evalEpisodes: 10,
);
```

See [Samples](samples.md) for complete executable programs.
