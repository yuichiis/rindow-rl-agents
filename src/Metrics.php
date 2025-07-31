<?php
namespace Rindow\RL\Agents;

interface Metrics
{
    public function attract(array $metrics) : void;
    public function isAttracted(string $name) : bool;
    public function attracted() : array;
    public function update(string $name, float $value) : void;
    public function result(string $name) : float;
    public function reset(string $name) : void;
    public function add(string $name, float $value);
    public function resetAll() : void;
    public function record() : void;
    public function history() : array;
    public function render(?array $exclude=null) : string;
}
