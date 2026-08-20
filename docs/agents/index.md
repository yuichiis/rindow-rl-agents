# Agents

| Agent | Action space | Policy | Data |
|---|---|---|---|
| [A2C](a2c.md) | Discrete or continuous | Stochastic actor-critic | On-policy rollout |
| [DDPG](ddpg.md) | Continuous | Deterministic | Replay buffer |
| [DQN / DDQN](dqn.md) | Discrete | Epsilon-greedy Q policy | Replay buffer |
| [PPO](ppo.md) | Discrete or continuous | Clipped stochastic policy | On-policy rollout |
| [Q-learning](q-learning.md) | Discrete | Epsilon-greedy linear Q | Transition |
| [REINFORCE](reinforce.md) | Discrete | Categorical policy | Complete episode |
| [SAC+gSDE](sac-gsde.md) | Continuous | Entropy-regularized gSDE | Replay buffer |
| [True Online Sarsa(λ)](true-online-sarsa-lambda.md) | Discrete | Epsilon-greedy linear Q | Transition |

All neural agents receive a Rindow Neural Networks `Builder`. Linear tile-coded
agents receive the active linear algebra object and may optionally receive the
builder for device-to-host observation conversion.
