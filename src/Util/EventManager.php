<?php
namespace Rindow\RL\Agents\Util;

use Rindow\RL\Agents\EventManager as EventManagerInterface;

class EventManager implements EventManagerInterface
{
	protected array $events = [];

    protected function doNotify(callable $listener,array $parameters) : void
    {
        $listener($parameters);
    }

    public function attach(string $eventName, callable $listener) : void
    {
    	if(!isset($this->events[$eventName])) {
            $this->events[$eventName] = [];
        }
        $this->events[$eventName][] = $listener;
    }

    protected function select(string $eventName) : array
    {
        if(!isset($this->events[$eventName])) {
            return [];
        }
        return $this->events[$eventName];
    }

    public function notify(
        string $eventName,
        array $parameters = null,
    ) : void
    {
        if($parameters===null) {
            $parameters = [];
        }
        $queue = $this->select($eventName);
        foreach ($queue as $listener) {
            $this->doNotify($listener,$parameters);
        }
    }
}