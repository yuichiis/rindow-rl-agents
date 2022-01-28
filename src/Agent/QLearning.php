<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class QLearning extends Sarsa
{
    protected function getHistory($experience)
    {
        return $experience->recently(1);
    }

    protected function tdError(
        NDArray $q,NDArray $nextValues,float $reward,$history) : NDArray
    {
        $la = $this->la;
        //  TD = R(t+1)+gamma*max(Q(s(t+1),?))-Q(s(t),a(t))
        $imaxQ = $la->imax($nextValues);
        $maxQ = $nextValues[[$imaxQ,$imaxQ]];
        $td = $la->axpy($q,$la->increment(
            $la->scal($this->gamma,$la->copy($maxQ)),$reward),
            -1.0);
        return $td;
    }
}
