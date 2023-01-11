<?php
namespace Rindow\RL\Agents;

/**
 *
 */
interface Agent
{
    public function initialize(); // : Operation

    public function policy();

    public function register(EventManager $eventManager=null) : void;

    /**
    * @param  mixed $states : N x StatesDims typeof int or NDArray or array of int or array of NDArray
    * @return mixed $action : N x ActionDims typeof int or NDArray
    */
    public function action(mixed $observation, bool $training) : mixed;

    public function maxQValue(mixed $observation) : float;

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

    public function fileExists(string $filename) : bool;

    public function saveWeightsToFile(string $filename) : void;

    public function loadWeightsFromFile(string $filename) : void;
}
