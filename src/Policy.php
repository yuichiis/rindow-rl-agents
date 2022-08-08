<?php
namespace Rindow\RL\Agents;

use Interop\Polite\Math\Matrix\NDArray;

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
    * @param NDArray $values
    * @return int $action
    */
    public function action($values, bool $training, int $time=null);
}
