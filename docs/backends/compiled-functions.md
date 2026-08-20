# Compiled Functions

Rindow Neural Networks can record forward and backward operations as a compiled
function pipeline. The recorded graph refers to the NDArrays that existed when
the function was compiled.

Consequently, weight synchronization and checkpoint loading copy values into
existing buffers:

```php
$la->copy($source->value(), $target->value());
```

Replacing a `Variable` value with `assign()` after compilation would not retarget
the already recorded pipeline. The agents therefore use in-place copies for hard
target synchronization, Polyak updates, and loading built weights. Assignment is
only appropriate while initializing an unbuilt variable, before a compiled graph
can reference it.
