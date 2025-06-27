<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Agent
{
    public function initialize() : void; // : Operation

    public function policy() : ?Policy;

    public function register(?EventManager $eventManager=null) : void;

    /**
    * @param  NDArray $states : N x StatesDims typeof NDArray
    * @return NDArray $actions : N x ActionDims typeof NDArray
    */
    public function action(array|NDArray $states, ?bool $training=null, ?array $info=null) : NDArray;

    //public function maxQValue(NDArray $state) : float;

    /**
    * @param iterable $experience
    * @return float $loss
    */
    public function update(ReplayBuffer $experience) : float;

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
