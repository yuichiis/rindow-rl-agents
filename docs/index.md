# Rindow RL Agents Reference

Rindow RL Agents is a collection of reinforcement-learning agents, training
runners, replay storage, and device-aware environment adapters for Rindow
Neural Networks. The implementation supports both host and accelerator linear
algebra backends.

## Contents

- [Getting started](getting-started.md)
- [Core concepts](concepts.md)
- [Configuration reference](configuration.md)
- [Agents](agents/index.md)
- [Buffers](buffers/index.md)
- [Training runners](runners/index.md)
- [Environment adapters](environments/index.md)
- [Utilities](utilities/action-masks.md)
- [Checkpoint persistence](persistence/checkpoints.md)
- [CPU and GPU backends](backends/cpu-and-gpu.md)
- [Samples](samples.md)
- [Testing](testing.md)
- [API index](api-index.md)

## Package namespace

Public classes use the `Rindow\RL\Agents` namespace. Agent implementations are
under `Rindow\RL\Agents\Agent`, buffers under `Rindow\RL\Agents\ReplayBuffer`,
and device wrappers under `Rindow\RL\Agents\Env`.

This manual documents the source tree as it exists in this repository. It is an
API reference, not a guarantee that one hyperparameter set will converge for
every environment or random seed.
