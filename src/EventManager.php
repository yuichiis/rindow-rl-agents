<?php
namespace Rindow\RL\Agents;

interface EventManager
{
    public function attach(string $eventName, callable $callback) : void;
    public function notify(string $event, array $args = null) : void;
}
