<?php
namespace Rindow\RL\Agents\Distribution;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Distribution;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;

abstract class AbstractDistribution implements Distribution
{
    protected Builder $nn;
    protected object $g;

    public function __construct(
        Builder $nn,
    )
    {
        $this->nn = $nn;
        $this->g = $nn->gradient();
    }

    /**
     *   Return actions according to the probability distribution.
     * 
     */
    public function get_actions(
        ?bool $deterministic=null,
        ) : Variable
    {
        echo "func get_actions()\n";
        $deterministic ??= false;
        if($deterministic) {
            return $this->mode();
        }
        return $this->sample();
    }

    /**
     *   Continuous actions are usually considered to be independent,
     *   so we can sum components of the ``log_prob`` or the entropy.
     *
     *   :param tensor: shape: (n_batch, n_actions) or (n_batch,)
     *   :return: shape: (n_batch,) for (n_batch, n_actions) input, scalar for (n_batch,) input
     * 
     */
    public function sum_independent_dims(NDArry $tensor) : NDArray
    {
        echo "func sum_independent_dims()\n";
        $g = $this->g;
        if($tensor->ndim()> 1) {
            $tensor = $g->reduceSum($tensor, axis:1);
        } else {
            $tensor = $g->reduceSum($tensor);
        }
        return $tensor;
    }
}