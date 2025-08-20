<?php
namespace Rindow\RL\Agents\Agent\Sarsa;

use Interop\Polite\Math\Matrix\NDArray;
use function Rindow\Math\Matrix\R;
use Rindow\RL\Agents\Agent\QLearning\QLearning;
use Rindow\RL\Agents\ReplayBuffer;

class Sarsa extends QLearning
{
    protected function getHistory(ReplayBuffer $experience) : array
    {
        return $experience->get(-2);
    }

    protected function tdError(
        NDArray $q,
        NDArray $nextValues,
        float $reward,
        ?NDArray $nextMask,
        ReplayBuffer $experience,
        ) : NDArray
    {
        $la = $this->la;
        [$dmy0,$nextAction,$dmy1,$dmy2,$dmy3,$dmy4] = $experience->get(-1);
        // No masking is required as it uses the actual selected action.
        $nextQ = $la->gatherb($nextValues,$nextAction,axis:-1);
        $td = $la->axpy($q,$la->increment(
            $la->scal($this->gamma,$la->copy($nextQ)),$reward),
            -1.0);
        return $td;
    }

}
