<?php
namespace Rindow\RL\Agents;

/**
 *
 */
interface Agent
{
    public function initialize(); // : Operation

    public function policy();

    public function setElapsedTime($elapsedTime) : void;
    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($observation,bool $training);

    public function getQValue($observation) : float;

    /**
    * @param iterable $experience
    * @return float $loss
    */
    public function update($experience) : float;

    /**
    * @return bool $stepUpdate
    */
    public function isStepUpdate() : bool;

    /**
    * @return bool $stepUpdate
    */
    public function subStepLength() : int;

    public function startEpisode(int $episode) : void;

    public function endEpisode(int $episode) : void;

    public function fileExists(string $filename) : bool;

    public function saveWeightsToFile(string $filename) : void;

    public function loadWeightsFromFile(string $filename) : void;
}
