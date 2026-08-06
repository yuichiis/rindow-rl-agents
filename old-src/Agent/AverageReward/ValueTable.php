<?php
namespace Rindow\RL\Agents\Agent\AverageReward;

use Rindow\RL\Agents\Estimator\AbstractTable;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class ValueTable extends AbstractTable
{
    /**
     * $rules: (numStates,numActions)
     */
    public function __construct(
        object $la,
        int $numActions,
        )
    {
        parent::__construct($la, numStates:1, numActions:$numActions);
        // initialize legal actions probabilities
        $this->table = $la->alloc([1,$numActions],dtype:NDArray::float32);
        //$this->probabilities = null;
        $this->initialize();
    }

    public function initialize() : void // : Operation
    {
        $la = $this->la;
        // initialize value table
        $la->zeros($this->table);
    }
}
