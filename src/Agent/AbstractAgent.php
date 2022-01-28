<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Agent;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

abstract class AbstractAgent implements Agent
{
    protected $elapsedTime;

    public function policy()
    {
        return $this->policy;
    }

    public function setElapsedTime($elapsedTime) : void
    {
        $this->elapsedTime = $elapsedTime;
    }
}
