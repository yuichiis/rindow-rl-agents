<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\EventManager;
use InvalidArgumentException;

abstract class AbstractAgent implements Agent
{
    protected $la;
    protected $policy;

    public function __construct(object $la,
        Policy|NDArray $policy=null, EventManager $eventManager=null)
    {
        $this->la = $la;
        $this->policy = $policy;
    }

    public function register(EventManager $eventManager=null) : void
    {
        $policy = $this->policy;
        if($policy instanceof Policy) {
            $policy->register($eventManager);
        }
    }

    public function policy()
    {
        return $this->policy;
    }
}
