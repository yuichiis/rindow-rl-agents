<?php
namespace Rindow\RL\Agents\Distribution;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\RL\Agents\Distribution;

class Normal implements Distribution
{
    protected Builder $nn;
    protected object $g;
    protected Variable $loc;
    protected Variable $stableScale;
    protected float $epsilon = 1e-8;
    protected Variable $hlog2pi;

    public function __construct(
        Builder $nn,
        Variable $loc,
        Variable $scale,
        )
    {
        $this->nn = $nn;
        $g = $nn->gradient();
        $this->g = $g;
        $this->loc = $loc;
        $this->stableScale = $g->add($scale,$this->epsilon);
        $this->hlog2pi = $g->constant(0.5 * log(2.0 * M_PI));
    }

    public function logProb(Variable $value) : Variable
    {
        $g = $this->g;
        $logProb = $g->sub(
            $g->sub(
                $g->scale(-0.5,$g->square($g->div($g->sub($value,$this->loc),$this->stableScale))),
                $g->log($this->stableScale)
            ),
            $this->hlog2pi,
        );
        return $logProb;
    }

    public function entropy() : Variable
    {
        $g = $this->g;
        $entropy = $g->add($g->add(0.5, $this->hlog2pi), $g->log($this->stableScale));
        $entropy = $g->add($g->zerosLike($this->loc),$entropy); // 他のテンソルとの互換性のため
        return $entropy;
    }

    public function mean() : Variable
    {
        return $this->loc;
    }

    public function scale() : Variable
    {
        return $this->stableScale;
    }

    public function sample(?array $batchShape=null) : Variable
    {
        $g = $this->g;
        $sample = $g->add(   // (batchSize,numActions)
            $this->loc,
            $g->mul(
                $this->stableScale,
                $g->randomNormal($this->loc,batchShape:$batchShape),
            )
        );
        return $sample;
    }
}
