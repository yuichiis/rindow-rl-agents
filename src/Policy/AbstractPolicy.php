<?php
namespace Rindow\RL\Agents\Policy;

use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\EventManager;

abstract class AbstractPolicy implements Policy
{
    protected $la;

    public function __construct($la)
    {
        $this->la = $la;
    }

    public function register(EventManager $eventManager=null) : void
    {
    }
}