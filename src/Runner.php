<?php
namespace Rindow\RL\Agents;

/**
 *
 */

interface Runner
{
    const EVENT_START_EPISODE = 'start-episode';
    const EVENT_END_EPISODE   = 'end-episode';
    const EVENT_START_STEP    = 'start-step';
    const EVENT_END_STEP      = 'end-step';
    const EVENT_START_ROLLOUT = 'start-rollout';
    const EVENT_END_ROLLOUT   = 'end-rollout';

    public function agent() : Agent;

    public function train(
        ?int $numIterations=null, ?int $numRolloutSteps=null, ?int $maxSteps=null, ?array $metrics=null,
        ?int $evalInterval=null, ?int $numEvalEpisodes=null, ?int $logInterval=null,
        ?int $targetScore=null, ?int $numAchievements=null,
        ?int $verbose=null
    ) : array;
}
