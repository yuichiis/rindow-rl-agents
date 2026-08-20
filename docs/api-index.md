# API Index

## Agents and models

- `A2C\A2CAgent`, `A2C\ActorCritic` — [A2C](agents/a2c.md)
- `DDPG\DDPGAgent`, `DDPG\Actor`, `DDPG\Critic` — [DDPG](agents/ddpg.md)
- `DDPG\OrnsteinUhlenbeckNoise` — [DDPG exploration](agents/ddpg.md)
- `DQN\DQNAgent`, `DQN\QNetwork` — [DQN](agents/dqn.md)
- `PPO\PPOAgent`, `PPO\ActorCritic` — [PPO](agents/ppo.md)
- `QLearning\QLearningAgent` — [Q-learning](agents/q-learning.md)
- `Reinforce\ReinforceAgent`, `Reinforce\PolicyNetwork` — [REINFORCE](agents/reinforce.md)
- `SAC\SACGSDEAgent`, `SAC\GSDEActor`, `SAC\Critic`, `SAC\QNetwork` — [SAC+gSDE](agents/sac-gsde.md)
- `Sarsa\TrueOnlineSarsaLambdaAgent`, `Sarsa\TileCoder` — [True Online Sarsa(λ)](agents/true-online-sarsa-lambda.md)

## Storage

- `ReplayBuffer\ReplayBuffer` — [ReplayBuffer](buffers/replay-buffer.md)
- `ReplayBuffer\RolloutBuffer` — [RolloutBuffer](buffers/rollout-buffer.md)

## Runners

Each agent namespace contains a `Runner`. See [runner overview](runners/index.md),
[on-policy](runners/on-policy-runners.md), and
[off-policy](runners/off-policy-runners.md).

## Environment wrappers

- `Env\CartPole\DeviceWrapper`
- `Env\MountainCar\DeviceWrapper`
- `Env\ContinuousMountainCar\DeviceWrapper`
- `Env\Pendulum\DeviceWrapper`
- `Env\Maze\DeviceWrapper`

See [Device wrappers](environments/device-wrappers.md).

## Utilities

- `Util\ActionMask` — [Action masks](utilities/action-masks.md)
- `Util\GradientClipping` — [Gradient clipping](utilities/gradient-clipping.md)
- `Util\TensorValidation` — [Tensor validation](utilities/tensor-validation.md)
- `Util\ProgressBar` — [Progress reporting](utilities/progress-reporting.md)
