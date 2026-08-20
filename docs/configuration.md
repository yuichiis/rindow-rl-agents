# Configuration Reference

The values below are constructor defaults, not universal recommendations.

| Agent | Learning rate | Discount | Batch/rollout | Main stabilization |
|---|---:|---:|---:|---|
| A2C | `7e-4` | runner `0.99` | rollout `5` | GAE, global gradient clipping |
| DDPG | actor `1e-4`, critic `1e-3` | `0.99` | `128` | target networks, Polyak `0.005` |
| DQN | `1e-3` | `0.99` | `64` | replay, target update every `500` |
| PPO | `3e-4` | runner `0.99` | `64`, 10 epochs | ratio clip `0.2`, value clipping |
| Q-learning | `0.3` | `0.99` | transition | tile coding, epsilon `0.05` |
| REINFORCE | `1e-2` | runner `0.99` | episode | optional return normalization |
| SAC+gSDE | required arguments | required | required | twin Q, entropy tuning, Polyak |
| True Online Sarsa(λ) | `0.3` | `1.0` | transition | Dutch traces, lambda `0.9` |

## Common parameters

- `hiddenLayers`: dense hidden-layer widths.
- `featureLayers`: cloned feature-extractor templates. Each online and target
  model receives independent layer instances.
- `maxGradNorm`: global gradient norm limit; `INF` disables clipping.
- `solvedReward`: evaluation threshold. `null` disables early success detection.
- `solvedEvaluations`: required consecutive successful evaluations.
- `rewardFunction`: training-reward callback.
- `observationFunction`: callback converting raw observations to network input.
- `bestModelFile`: runner destination for the best evaluated model.

## Environment overrides in samples

Samples commonly read `RL_*` environment variables. These are sample controls,
not agent API. Inspect the selected sample for its accepted names and defaults.
