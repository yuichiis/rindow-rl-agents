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
        return $experience->recently(2);
    }

    protected function tdError(
        NDArray $q,
        NDArray $nextValues,
        float $reward,
        ?array $info,
        iterable $history,
        ) : NDArray
    {
        $la = $this->la;
        [$dmy0,$nextAction,$dmy1,$dmy2,$dmy3,$dmy4] = $history[1];
        // R(t+1)+gamma*Q(s(t+1),a(t+1))-Q(s(t),a(t))
        //$nextAction = $la->scalar($nextAction);
        //$nextQ = $nextValues[R($nextAction,$nextAction+1)];
        //$nextQ = $la->gatherb($nextValues,$nextAction);
        //$nextValues = $la->nan2num($la->copy($nextValues),alpha:-INF);
        if($info!=null) {
            $masks = $this->extractMasks([$info]);
            //$nextValues = $la->nan2num($la->copy($nextValues),alpha:-INF);
            $nextValues = $la->masking($masks,$la->copy($nextValues),fill:-INF);
        }
        $nextQ = $la->reduceMax($nextValues,axis:-1);
        $td = $la->axpy($q,$la->increment(
            $la->scal($this->gamma,$la->copy($nextQ)),$reward),
            -1.0);
        return $td;
    }

}
