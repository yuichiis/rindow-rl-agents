# Testing

Run the PHPUnit suite from the repository root:

```powershell
phpunit -c phpunit.xml
```

Unit tests use fixed tensors and small networks. They cover replay layouts, GAE,
termination versus truncation, action masks, global gradient clipping, tile
coding, exact tabular TD updates, DQN versus DDQN targets, actor-critic losses,
action bounds, Polyak updates, checkpoint round trips, and in-place target buffer
identity.

Convergence is intentionally outside the unit-test contract. Learning depends on
random seeds, environment dynamics, reward design, backend arithmetic, and
hyperparameters. Sample PowerShell scripts under `tests/` are experiment drivers
and smoke tests rather than deterministic PHPUnit cases.

When adding an algorithm test, prefer a one- or two-transition analytical
fixture. Assert exact targets where possible and use tolerances for floating
point results. Keep CPU/GPU statistical tests separate from fast unit tests.
