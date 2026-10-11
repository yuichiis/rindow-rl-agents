Rindow RL Agents
================

Reinforcement-learning agents and training utilities for
[Rindow Neural Networks](https://github.com/rindow/rindow-neuralnetworks).

Overview
--------

This package provides discrete and continuous control algorithms, replay and rollout buffers, training execution capabilities, action mask support, checkpoint saving functionality, and CPU/GPU-compatible tensor operations.

Website:
- Rindow project: [https://rindow.github.io/](https://rindow.github.io/)
- Rindow Neural Networks: [https://rindow.github.io/neuralnetworks/reinforcement-learning.html](https://rindow.github.io/neuralnetworks/reinforcement-learning.html)
- Documentation: [https://rindow.github.io/rl-agents/](https://rindow.github.io/rl-agents/)

Features
--------

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

Linked library
--------------
- Rindow Neural Networks: Machine Learning & Deep Learning library for PHP
- Rindow Math Matrix: Scientific calculation library
- Rindow Matlib: A fast matrix calculation library suitable for machine learning
- OpenBLAS: Fast Matrix Arithmetic Library
- Rindow Math Plot: Visualize machine learning results
- OpenCL: GPU computational programming interface
- CLBlast: High-speed matrix calculation library using OpenCL

Required environment
--------------------

- PHP 8.1 or later
- Windows, Linux, macOS

Installation
------------

### 1. Install OpenBLAS

For Windows, download the binary and set the execution path.
- https://github.com/OpenMathLib/OpenBLAS/releases

For Linux, install OpenBLAS using the package manager.
```bash
$ sudo apt install libopenblas0 liblapacke
```

For macOS, OpenBLAS is not required; the veclib included with macOS is used.

### 2. Install Rindow-Matlib

For Windows, download the binary and set the execution path.

- https://github.com/rindow/rindow-matlib/releases

For Linux, download and install Rindow-Matlib using the package manager.

- https://github.com/rindow/rindow-matlib/releases

```bash
$ sudo apt install ./rindow-matlib_X.X.X-XX.XX_amd64.deb
```

For macOS, download and install Rindow-Matlib using the package manager.

- https://github.com/rindow/rindow-matlib/releases

```bash
$ tar xvf rindow-matlib-X.X.X-Darwin-XXXX.tar.gz
$ sudo cp -r usr/include /usr/local/
$ sudo cp -r usr/lib /usr/local/
```

### 3. Install CLBlast (Optional)

For GPU acceleration, download and install CLBlast using the package manager or just extract the package.

- https://github.com/CNugteren/CLBlast/releases

The GPU driver is the standard Windows version, and the integrated GPU also functions.

For Linux, install CLBlast using the package manager.
```bash
$ sudo apt install clinfo intel-opencl-icd libclblast1
```

Linux standard OpenCL drivers include:

- mesa-opencl-icd
- beignet-opencl-icd
- intel-opencl-icd
- nvidia-opencl-icd-xxx
- pocl-opencl-icd

For macOS, the GPU is not supported.

### 4. Install Rindow-RL-Agent

Install the package with Composer:

```bash
$ composer require rindow/rindow-rl-agents
```

Composer installs Rindow RL Agents and its declared runtime dependencies. Your
application only needs the generated Composer autoloader:

```php
require __DIR__.'/vendor/autoload.php';
```

The exact PHP extensions and native libraries required for acceleration depend
on the selected Rindow Math Matrix backend. A CPU backend is sufficient to use
the package.

## Sample code execution

```powershell
PS1> php samples/cartpole-dqn.php
```


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
| True Online Sarsa(lambda) | Discrete | Per transition | On-policy learning with Dutch traces |

## CPU and GPU execution

Rindow Neural Networks selects the numerical backend. For example, an OpenCL
GPU backend can be selected before starting PHP:

```powershell
PS> $env:RINDOW_NEURALNETWORKS_BACKEND = "rindowclblast::GPU"
PS> php samples/cartpole-dqn.php
```

> The agent's backend `NDArray` objects are used for observations, replay storage, and action masks. The included device wrapper transfers observations from the environment to the configured device and returns actions to the host environment. Environment stepping and runner control flow are handled on the PHP host. Batched network operations, loss calculations, reductions, masking, replay data collection, and optimizer updates are executed on the selected backend.

> Executions on CPU and GPU do not necessarily follow identical stochastic trajectories. Rather than expecting bit-for-bit identical training logs, you should compare results across multiple seed values ​​or look at task-level statistics.


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

Some runners accept a `rewardFunction` callback; this modifies the reward stored for learning without altering the raw reward reported during evaluation. While this feature is useful for experiments involving controlled reward shaping, arbitrary shaping can alter the optimal policy.

Truncation due to time limits and true MDP termination are handled separately. Standard runners typically bootstrap upon truncation, whereas they use a bootstrap value of zero upon true termination.

## Checkpoints

Agents expose a common persistence pattern:

```php
$agent->saveWeightsToFile(__DIR__.'/agent.weights');
$agent->loadWeightsFromFile(__DIR__.'/agent.weights');
```

Neural network checkpoints include the online model and the target networks required to resume training; in the case of SAC, the learned entropy coefficient is also saved. Since existing model buffers are updated in-place, checkpointing and target synchronization remain compatible with compiled Rindow Neural Networks function pipelines.

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

For some samples, you can use `RL_*` environment variables to specify parameters such as the seed value, training duration, evaluation interval, model path, and demo controls. Please check the sections where constants or environment variables are referenced at the beginning of each sample.

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
vendor/bin/phpunit -c tests
```

On Windows:

```powershell
vendor\bin\phpunit.bat -c tests
```

This unit test suite verifies analytical updates, loss, GAE, action masking, buffer layout, action bounds, target network synchronization, Polyak updates, and checkpointing behavior. Training convergence is intentionally excluded from the scope of these unit tests (the contract). The scripts under `tests/` are intended for running experiments and smoke tests.

