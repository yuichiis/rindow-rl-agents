# Core Concepts

## Tensor ownership

An `NDArray` belongs to the linear-algebra backend that created it. Neural agents
expect observations on their configured backend. Device wrappers transfer raw
environment observations to the device and actions back to the host.

Model weights are updated in place. This preserves the `NDArray` identity used
by compiled Rindow Neural Networks functions; replacing a `Variable` value after
compilation would leave the compiled pipeline referring to the old buffer.

## Observation shapes

`obsDim` may be an integer or an array. An integer `4` means a single observation
has shape `[4]`; `[84,84,4]` describes an image observation. Agent inference adds
the leading batch dimension. Optional `featureLayers` can provide a CNN or RNN
feature extractor where supported.

Dictionary observations require `stateField`. If `actionMaskField` is also set,
that field must contain one value per discrete action and at least one action
must be enabled.

## Episode boundaries

The environment API returns both `terminated` and `truncated`:

- `terminated` means the MDP reached a terminal state; its bootstrap value is zero.
- `truncated` means an external limit ended the episode. Value-based algorithms
  normally bootstrap from the final observation.

`RolloutBuffer` records both concepts separately so GAE can stop across an
episode boundary while still bootstrapping a time-limit truncation.

## Raw and transformed rewards

Runner `rewardFunction` callbacks may transform rewards used for training.
Evaluation reports the raw environment reward, and runners with detailed
evaluation also report the transformed reward. A potential-based transform

```text
r'(s,a,s') = r(s,a,s') + gamma*Phi(s') - Phi(s)
```

preserves the optimal policy under its usual assumptions. Arbitrary shaping,
such as absolute velocity, can change the task being optimized.

## On-policy and off-policy data

A2C and PPO consume a finite `RolloutBuffer` collected by the current policy.
DQN, DDPG, and SAC sample with replacement from a `ReplayBuffer`. REINFORCE uses
complete episodes. Q-learning and Sarsa update directly from each transition.

## Training and evaluation actions

Training selection may use epsilon-greedy exploration, policy sampling, OU
noise, Gaussian noise, or gSDE. Evaluation methods are deterministic unless an
SAC runner is explicitly asked to evaluate with gSDE noise.
