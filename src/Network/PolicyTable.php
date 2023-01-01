<?php
namespace Rindow\RL\Agents\Network;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class PolicyTable extends QTable
{
    public function __construct($la, NDArray $rules)
    {
        $this->table = $la->alloc($rules->shape(),$rules->dtype());
        parent::__construct($la, $rules);
    }

    public function initialize() // : Operation
    {
        $la = $this->la;
        $la->copy($this->rules,$this->table);
        $this->rulesProbs = $this->generateProbabilities($this->rules);
    }
}