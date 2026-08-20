# Samples

Executable programs are under `samples/`.

| Environment | Algorithms |
|---|---|
| CartPole | A2C, DQN, PPO, Q-learning, REINFORCE, Sarsa |
| Maze | A2C, DQN, PPO, Q-learning, Sarsa |
| MountainCar | A2C, DQN, PPO, Q-learning, Sarsa |
| ContinuousMountainCar | A2C, DDPG, PPO+gSDE, SAC+gSDE |
| Pendulum | A2C, DDPG, PPO+gSDE, SAC+gSDE |

Some CartPole, Maze, ContinuousMountainCar, and Pendulum programs include image
observation variants with feature layers.

Run a sample from the repository root so relative model and graphics paths are
resolved correctly:

```powershell
php samples/cartpole-dqn.php
```

MountainCar samples expose alternative reward transforms for experiments. Raw
evaluation reward remains the criterion for solving the original environment.
Review each sample's constants and `RL_*` environment-variable helpers before
launching a long run.
