# Environments

The agents use `Interop\Polite\AI\RL\Environment`. Its core interaction methods
are:

```text
reset(seed?) -> [observation, info]
step(action) -> [observation, reward, terminated, truncated, info]
```

Bundled adapters cover CartPole, MountainCar, ContinuousMountainCar, Pendulum,
and Maze. See [Device wrappers](device-wrappers.md).
