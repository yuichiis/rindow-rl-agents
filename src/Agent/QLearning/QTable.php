<?php
namespace Rindow\RL\Agents\Agent\QLearning;

use Rindow\RL\Agents\Estimator\AbstractTable;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class QTable extends AbstractTable
{
    /**
     * $rules: (numStates,numActions)
     */
    public function __construct(
        object $la,
        int $numStates,
        int $numActions,
        )
    {
        parent::__construct($la, $numStates, $numActions);
        // initialize legal actions probabilities
        //$this->probabilities = $this->generateProbabilities($this->rules);
        $this->initialize();
    }

    public function initialize() : void // : Operation
    {
        $la = $this->la;
        // initialize Q table
        $this->table = $la->randomUniform([$this->numStates,$this->numActions],0,1);
        //$la->multiply($this->rules,$this->table);
    }
}
