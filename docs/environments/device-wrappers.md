# Device Wrappers

Device wrappers accept a Rindow Neural Networks builder and a host environment.
They forward metadata and rendering methods unchanged while transferring arrays
across the backend boundary.

For standard environments, `reset()` converts the observation with
`deviceArray()`. `step()` requires an `NDArray`, transfers the action with
`hostArray()`, calls the wrapped environment, and converts the next observation
back to the device.

The Maze wrapper handles dictionary observations:

```php
[
    'location' => NDArray,
    'actionMask' => NDArray,
]
```

Both entries are transferred independently. Invalid observations or non-NDArray
actions raise an exception early, avoiding accidental host/device mixing.

Wrappers also proxy `maxEpisodeSteps()`, `rewardThreshold()`, spaces, `render()`,
`show()`, `close()`, and context entry/exit.
