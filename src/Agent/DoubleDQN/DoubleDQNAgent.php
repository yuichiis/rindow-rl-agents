<?php
namespace Rindow\RL\Agents\Agent\DoubleDQN;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;

/**
 * Double DQN agent.
 *
 * The online network selects the bootstrap action while the target network
 * evaluates it. This avoids the maximization bias of selecting and evaluating
 * the next action with the same target network.
 */
class DoubleDQNAgent extends DQNAgent
{
    protected function nextStateValues(
        NDArray $nextObservations,
        ?NDArray $nextActionMasks,
    ) : NDArray {
        $onlineQ = $this->qNetwork->forward(
            $this->g->Variable($nextObservations),false
        )->value();
        if ($nextActionMasks !== null) {
            $onlineQ = $this->la->masking($nextActionMasks,$onlineQ,fill:-1.0e9);
        }
        $nextActions = $this->la->reduceArgMax(
            $onlineQ,axis:1,dtype:NDArray::int32
        );

        $targetQ = $this->targetNetwork->forward(
            $this->g->Variable($nextObservations),false
        )->value();
        return $this->la->gather($targetQ,$nextActions,axis:1);
    }
}
