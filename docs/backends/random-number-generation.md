# Random Number Generation

Agents use backend random operations for uniform exploration, categorical action
sampling, Gaussian policies, gSDE matrices, replay indices, and minibatch
permutations. Keeping random generation on the active backend avoids a host
round trip in hot paths.

The OpenCL math driver in this workspace uses PCG32-based per-work-item streams
for uniform and normal generation. Normal samples are derived from uniform
samples with a Gaussian transform. Separate calls and backend choices may still
consume random streams differently, so a common seed provides reproducibility
within a configuration, not CPU/GPU identity.

For robust experiments, run multiple seeds and report their distribution. A
single successful or failed reinforcement-learning run is not a backend test.
