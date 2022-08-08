<?php
namespace Rindow\RL\Agents;

use Countable;
/**
 *
 */
interface ReplayBuffer extends Countable
{
    public function size() : int;
    public function count() : int;
    public function add($item) : void;
    public function last();
    public function recently(int $quantity) : iterable;
    public function sample(int $quantity) : iterable;
}
