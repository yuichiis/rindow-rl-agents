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

    public function getQValue($observation);

    /**
    * @param iterable $experience
    */
    public function update($experience);

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
}
