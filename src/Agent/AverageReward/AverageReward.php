<?php
namespace Rindow\RL\Agents\Agent\AverageReward;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\AbstractAgent;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

class AverageReward extends AbstractAgent
{
    protected ValueTable $valueTable;
    protected int $numActions;
    protected NDArray $numTrials;

    public function __construct(
        object $la,
        int $numActions,
        ?Policy $policy=null,
        ?EventManager $eventManager=null)
    {
        $this->valueTable = new ValueTable($la,numActions:$numActions);
        $this->numActions = $numActions;
        parent::__construct($la,$policy,$eventManager);
        $this->initialize();
    }

    public function initialize() : void // : Operation
    {
        $la = $this->la;
        $this->valueTable->initialize();
        $this->numTrials = $la->zeros($la->alloc([$this->numActions],NDArray::float32));
    }

    public function resetData() : void
    {
        $this->valueTable->initialize();
    }

    public function isStepUpdate() : bool
    {
        return false;
    }

    public function subStepLength() : int
    {
        return 1;
    }

    public function numRolloutSteps() : int
    {
        return 1;
    }

    protected function estimator() : Estimator
    {
        return $this->valueTable;
    }

    /**
    * @param Any $params
    * @return Any $action
    */
    public function update($experience) : float
    {
        $la = $this->la;
        //$backupValue = $la->copy($this->values);
        //$backupNumTrials = $la->copy($this->numTrials);

        [$state,$action,$nextState,$reward,$done,$truncated,$info] = $experience->last();
        $actionNumber = $la->scalar($action);
        if($actionNumber<0 || $actionNumber>=$this->numActions) {
            throw new InvalidArgumentException('Invalid Action');
        }
        $table = $this->estimator()->table(); // table = (1,numActions)

        $n = $this->numTrials[R($actionNumber,$actionNumber+1)];
        $v = $table[0][R($actionNumber,$actionNumber+1)];

        $la->increment($n,1.0);

        // V(t) = ((n-1)/n)*V(t-1) + 1/n*R(t)
        //      =   (1-1/n)*V(t-1) + 1/n*R(t)
        $la->multiply(
            $la->increment($la->reciprocal($la->copy($n)),1.0,-1.0),
            $v
        );
        $backupTmpV = $la->copy($v);
        $la->axpy(
            $la->scal($reward,$la->reciprocal($la->copy($n))),
            $v
        );

        return 0.0;
    }

    public function fileExists(string $filename) : bool
    {
        return $this->valueTable->fileExists($filename);
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        $this->valueTable->setPortableSerializeMode($mode);
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $this->valueTable->saveWeightsToFile($filename);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $this->valueTable->loadWeightsFromFile($filename);
    }
}
