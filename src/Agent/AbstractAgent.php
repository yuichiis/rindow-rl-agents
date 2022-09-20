<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Policy;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

abstract class AbstractAgent implements Agent
{
    protected $elapsedTime;
    protected $policy;

    protected function setPolicy(Policy|NDArray $policy)
    {
        $this->policy = $policy;
    }

    public function policy()
    {
        return $this->policy;
    }

    public function setElapsedTime($elapsedTime) : void
    {
        $this->elapsedTime = $elapsedTime;
    }
}
