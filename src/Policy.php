<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\EventManager;

/**
 *
 */
interface Policy
{
    /**
    * @return void
    */
    public function initialize();

    /**
    * @return void
    */
    public function register(EventManager $eventManager=null) : void;

    /**
    * @param NDArray $values   : N x ValuesDims typeof float32 ( ex. [[0.0,1.0],[0.5,1.5],[1.0,2.0]] )
    * @return NDArray $actions : N x ActionsDims typeof int32 of float32 ( ex. [[0],[1],[2]] )
    */
    public function action(QPolicy $qPolicy, NDArray $values, bool $training) : NDArray;
}
