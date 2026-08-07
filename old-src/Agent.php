<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment as Env;

/**
 *
 */
interface Agent
{
    public function initialize() : void; // : Operation

    public function policy() : ?Policy;

    public function register(?EventManager $eventManager=null) : void;

    public function setMetrics(Metrics $metrics) : void;

    public function metrics() : Metrics;

    public function resetData() : void;

    /**
    * @param  NDArray $states : N x StatesDims typeof NDArray
    * @return NDArray $actions : N x ActionDims typeof NDArray
    */
    public function action(array|NDArray $states, ?bool $training=null, ?array $info=null, ?bool $parallel=null) : NDArray;

    public function reset(Env $env) : array;

    public function step(Env $env, int $episodeSteps, NDArray $states, ?array $info=null) : array;

    public function collect(
        Env $env,
        ReplayBuffer $experience,
        int $step,
        int $episodeSteps,
        NDArray $states,
        ?array $info,
        ) : array;

    /**
    * @param iterable $experience
    * @return float $loss
    */
    public function update(ReplayBuffer $experience) : float;

    /**
    * @return bool $stepUpdate
    */
    public function isStepUpdate() : bool;

    public function subStepLength() : int;

    public function numRolloutSteps() : int;

    public function fileExists(string $filename) : bool;

    public function saveWeightsToFile(string $filename) : void;

    public function loadWeightsFromFile(string $filename) : void;
}
