# Rindow RL Agents

Reinforcement-learning agents and training utilities for
[Rindow Neural Networks](https://github.com/rindow/rindow-neuralnetworks).

The package provides discrete and continuous-control algorithms, replay and
rollout buffers, training runners, action-mask support, checkpoint persistence,
and CPU/GPU-compatible tensor operations.

## Features

- Advantage Actor-Critic (A2C)
- Deep Q-Network and Double DQN (DQN/DDQN)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Soft Actor-Critic with generalized State-Dependent Exploration (SAC+gSDE)
- REINFORCE
- Linear Q-learning with tile coding
- True Online Sarsa(λ) with tile coding
- Discrete action masks
- Generalized Advantage Estimation (GAE)
- Replay and on-policy rollout buffers
- Best-model checkpoints and deterministic evaluation
- Host and OpenCL accelerator backends

## Installation

Install the package with Composer:

```bash
composer require rindow/rindow-rl-agents
```

Composer installs Rindow RL Agents and its declared runtime dependencies. Your
application only needs the generated Composer autoloader:

```php
require __DIR__.'/vendor/autoload.php';
```

The exact PHP extensions and native libraries required for acceleration depend
on the selected Rindow Math Matrix backend. A CPU backend is sufficient to use
the package.

## Quick start: CartPole with DDQN

The following example assumes that a CartPole environment implementing
`Interop\Polite\AI\RL\Environment` is available. The environment and evaluation
environment are separate so evaluation does not disturb the training trajectory.

```php
<?php
require __DIR__.'/vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;
use Rindow\RL\Agents\Agent\DQN\Runner;
use Rindow\RL\Agents\Env\CartPole\DeviceWrapper;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLA = $mo->laRawMode();
$la->setSeed(42);

$env = new CartPoleV1($hostLA);
$evalEnv = new CartPoleV1($hostLA);

if ($la->accelerated()) {
    $env = new DeviceWrapper($nn, $env);
    $evalEnv = new DeviceWrapper($nn, $evalEnv);
}

$obsDim = $env->observationSpace()->shape()[0];
$numActions = $env->actionSpace()->n();

$agent = new DQNAgent(
    $nn,
    obsDim: $obsDim,
    numActions: $numActions,
    hiddenLayers: [128, 128],
    learningRate: 1.0e-3,
    gamma: 0.99,
    batchSize: 64,
    targetUpdateInterval: 250,
    ddqn: true,
);

$runner = new Runner(
    $la,
    $env,
    $evalEnv,
    $agent,
    obsDim: $obsDim,
    bufferSize: 100_000,
    solvedReward: 475.0,
    solvedEvaluations: 3,
);

$history = $runner->train(
    totalSteps: 200_000,
    learningStarts: 1_000,
    trainEvery: 4,
    evalEvery: 2_000,
    evalEpisodes: 30,
    epsilonStart: 1.0,
    epsilonEnd: 0.05,
    epsilonDecaySteps: 50_000,
    bestModelFile: __DIR__.'/cartpole-ddqn.weights',
);
```

Reinforcement learning is stochastic. Treat the values above as a complete API
example, not as a convergence guarantee for every backend or random seed.

## Choosing an algorithm

| Algorithm | Action space | Data model | Typical use |
|---|---|---|---|
| A2C | Discrete or continuous | On-policy rollout | Compact synchronous actor-critic |
| DQN/DDQN | Discrete | Replay buffer | Discrete value-based control |
| PPO | Discrete or continuous | On-policy rollout | Clipped policy optimization |
| DDPG | Continuous | Replay buffer | Deterministic continuous control |
| SAC+gSDE | Continuous | Replay buffer | Entropy-regularized continuous control |
| REINFORCE | Discrete | Complete episode | Basic Monte Carlo policy gradient |
| Q-learning | Discrete | Per transition | Linear tile-coded value learning |
| True Online Sarsa(λ) | Discrete | Per transition | On-policy learning with Dutch traces |

## CPU and GPU execution

Rindow Neural Networks selects the numerical backend. For example, an OpenCL
GPU backend can be selected before starting PHP:

```powershell
$env:RINDOW_NEURALNETWORKS_BACKEND = "rindowclblast::GPU"
php samples/cartpole-dqn.php
```

Use the agent backend's `NDArray` objects for observations, replay storage, and
action masks. The bundled device wrappers transfer environment observations to
the configured device and actions back to the host environment.

Environment stepping and runner control flow remain on the PHP host. Batched
network operations, losses, reductions, masking, replay gathers, and optimizer
updates execute on the selected backend.

CPU and GPU runs need not follow identical stochastic trajectories. Compare
multiple seeds and task-level statistics rather than expecting bitwise-identical
training logs.

## Action masks

DQN, A2C, PPO, Q-learning, and True Online Sarsa(λ) support discrete action
masks. Configure dictionary fields when creating the agent:

```php
$agent = new DQNAgent(
    $nn,
    obsDim: 2,
    numActions: 4,
    stateField: 'location',
    actionMaskField: 'actionMask',
);
```

Each mask must contain one value per action and enable at least one action.
Disabled actions are excluded during both exploration and greedy selection.

## Reward transformation

Several runners accept a `rewardFunction` callback. It changes the reward stored
for training without changing the raw reward reported by evaluation. This is
useful for controlled reward-shaping experiments, but arbitrary shaping can
change the optimal policy.

Time-limit truncation and true MDP termination are handled separately. The
bundled runners normally bootstrap at truncation and use zero bootstrap value at
true termination.

## Checkpoints

Agents expose a common persistence pattern:

```php
$agent->saveWeightsToFile(__DIR__.'/agent.weights');
$agent->loadWeightsFromFile(__DIR__.'/agent.weights');
```

Neural checkpoints include the online models and any target networks needed to
resume training. SAC also stores its learned entropy coefficient. Existing model
buffers are updated in place so checkpoints and target synchronization remain
compatible with compiled Rindow Neural Networks function pipelines.

## Samples

The `samples/` directory contains complete programs for:

- CartPole
- Maze
- MountainCar
- Continuous MountainCar
- Pendulum

It includes vector and image-observation examples where applicable. Run samples
from the project root:

```bash
php samples/cartpole-ppo.php
```

Some samples accept `RL_*` environment variables for seeds, training length,
evaluation intervals, model paths, and demo control. See the constants and
environment lookups at the top of each sample.

## Documentation

The full reference manual starts at [docs/index.md](docs/index.md).

- [Getting started](docs/getting-started.md)
- [Core concepts](docs/concepts.md)
- [Configuration reference](docs/configuration.md)
- [Agent reference](docs/agents/index.md)
- [Buffers](docs/buffers/index.md)
- [Training runners](docs/runners/index.md)
- [CPU and GPU backends](docs/backends/cpu-and-gpu.md)
- [API index](docs/api-index.md)

## Testing

Install development dependencies and run PHPUnit:

```bash
vendor/bin/phpunit -c phpunit.xml
```

On Windows:

```powershell
vendor\bin\phpunit.bat -c phpunit.xml
```

The unit suite checks analytical update targets, losses, GAE, action masks,
buffer layouts, action bounds, target-network synchronization, Polyak updates,
and checkpoint behavior. Learning convergence is intentionally kept outside the
unit-test contract; the scripts under `tests/` are experiment and smoke-test
drivers.

## Package

Composer package name: `rindow/rindow-rl-agents`

Primary namespace: `Rindow\RL\Agents`
