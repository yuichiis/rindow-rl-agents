# Checkpoints

Neural-agent methods use:

```php
$agent->saveWeightsToFile($path, portable: true);
$agent->loadWeightsFromFile($path);
```

Each serialized checkpoint contains a format identifier, version, relevant
dimensions, and all models needed to resume that algorithm. Target networks are
included for DQN, DDPG, and SAC; SAC also stores `logAlpha`.

Loading validates format, version, and dimensions before changing weights.
Existing built variables are updated by copying into their current NDArrays.
This is required for compiled functions, which retain references to the original
buffers. Unbuilt model variables may be initialized during loading.

Q-learning and Sarsa serialize their PHP weight tables. Sarsa deliberately
clears eligibility traces after loading because traces belong to an active
episode, not to the persistent value function.

SAC writes a temporary file first and then replaces the destination so an
interrupted write is less likely to destroy the previous checkpoint.

Portable checkpoints transfer data to host representation. Non-portable mode is
available through the underlying Rindow Neural Networks model format where
supported.
