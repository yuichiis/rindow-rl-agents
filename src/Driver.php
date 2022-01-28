<?php
namespace Rindow\RL\Agents;

/**
 *
 */

interface Driver
{
    public function agent() : Agent;

    public function train(
        $numIterations=null,$maxSteps=null,array $metrics=null,
        $evalInterval=null,$numEvalEpisodes=null,$logInterval=null,$verbose=null) : array;
}
