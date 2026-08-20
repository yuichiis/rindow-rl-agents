# Tensor Validation

Class: `Rindow\RL\Agents\Util\TensorValidation`

- `allFinite($la,$values): bool` returns true only when every element is finite.
- `hasInf($la,$values): bool` reports positive or negative infinity.

Both methods accept device NDArrays and normalize scalar reduction results from
CPU and accelerator backends. `allFinite()` also rejects NaN, while `hasInf()`
does not report NaN.
