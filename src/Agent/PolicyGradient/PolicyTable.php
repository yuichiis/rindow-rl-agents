<?php
namespace Rindow\RL\Agents\Agent\PolicyGradient;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator\AbstractTable;

class PolicyTable extends AbstractTable
{
    public function __construct(
        object $la,
        int $numStates,
        int $numActions,
        )
    {
        $this->table = $la->alloc([$numStates, $numActions],dtype:NDArray::float32);
        parent::__construct($la, $numStates, $numActions);
        $this->initialize();
    }

    public function initialize() : void // : Operation
    {
        $la = $this->la;
        $la->ones($this->table);
        //$la->masking($this->masks,$this->table);
        //$this->probabilities = $la->copy($this->table);
    }

}