# CPU and GPU Backends

Backend selection is handled by Rindow Math Matrix and Rindow Neural Networks.
The samples can select an OpenCL backend through the environment, for example:

```powershell
$env:RINDOW_NEURALNETWORKS_BACKEND = "rindowclblast::GPU"
php samples/cartpole-ppo.php
```

Use the agent's backend `primaryLA()` for observations, buffers, action masks,
and random tensors. Passing host NDArrays directly into a device model may force
conversion or fail, depending on the operation. Device wrappers make this
boundary explicit.

GPU execution does not imply that every control-flow operation is parallel.
Environment stepping and runner loops remain PHP-side; batched network, loss,
reduction, replay gather, masking, and optimizer operations run on the selected
backend.

CPU and GPU training are not expected to produce identical trajectories.
Floating-point reduction order and stochastic sampling differ. Tests should
assert equations, bounds, shapes, and finite values rather than bitwise equality.
