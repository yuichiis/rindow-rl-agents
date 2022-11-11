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
    * @param NDArray $values
    * @return int $action
    */
    public function action($values, bool $training);
}
