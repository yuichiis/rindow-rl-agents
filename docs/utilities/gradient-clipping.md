# Gradient Clipping

Class: `Rindow\RL\Agents\Util\GradientClipping`

```php
GradientClipping::clipByGlobalNorm(object $la, array $gradients,
                                   float $maxNorm): array
```

The function computes one norm across all gradient tensors. If it exceeds
`maxNorm`, every tensor is scaled by the same factor, preserving their relative
magnitudes. The operation mutates and returns the supplied NDArrays.

The accelerated path keeps per-gradient norms and the final scaling operation on
the device. The host path uses scalar norm accumulation. `INF` or an empty list
returns immediately; a non-positive bound uses the host implementation.
