<?php
namespace Rindow\RL\Agents\Distribution;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;

/**
 *   Bijective transformation of a probability distribution
 *   using a squashing function (tanh)
 *
 *   :param epsilon: small value to avoid NaN due to numerical imprecision.
 * 
 */
class TanhBijector
{
    protected Builder $nn;
    protected object $g;
    protected float $epsilon;

    public function __construct(Builder $nn, ?float $epsilon=null)
    {
        $this->nn = $nn;
        $this->g = $nn->gradient();
        $epsilon ??= 1e-6;
        $this->epsilon = $epsilon;
    }

    public function forward(Variable $x) : Variable
    {
        return $this->g->tanh($x);
    }

    /**
     *   Inverse of Tanh
     *   Taken from Pyro: https://github.com/pyro-ppl/pyro
     *   0.5 * torch.log((1 + x ) / (1 - x))
     * 
     */
    public function atanh(Variable $x) : Variable
    {
        $g = $this->g;
        return $g->scale(0.5,
            $g->sub(
                $g->log1p($x),
                $g->log1p($g->scale(-1,$x))
            )
        );
    }

    /**
     *   Inverse tanh.
     * 
     */
    public function inverse(Variable $y) : Variable
    {
        $g = $this->g;
        $eps = 1.192e-7; //$g->finfo($y->dtype())->eps;
        # Clip the action to avoid NaN
        return $this->atanh($g->clipByValue($y, (-1.0 + $eps), (1.0 - $eps)));
    }

    public function log_prob_correction(Variable $x) : Variable
    {
        $g = $this->g;
        # Squash correction (from original SAC implementation)
        return $g->log($g->add($g->sub(1.0, $g->square($g->tanh($x))), $this->epsilon));
    }
}
