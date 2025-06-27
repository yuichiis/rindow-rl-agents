<?php
namespace Rindow\RL\Agents\Agent\QLearning;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;
use function Rindow\Math\Matrix\R;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Agent\AbstractAgent;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;

class QLearning extends AbstractAgent
{
    protected QTable $valueTable;
    protected float $eta;
    protected float $gamma;

    public function __construct($la,
        int $numStates,
        int $numActions,
        Policy $policy,
        float $eta,
        float $gamma,
        ?EventManager $eventManager=null,
        ?object $mo=null,
        )
    {
        $table = $this->buildNetwork($la,$numStates, $numActions);
        $policy = $this->buildPolicy($la);
        parent::__construct($la,$policy,$eventManager);
        $this->valueTable = $table;
        $this->eta = $eta;
        $this->gamma = $gamma;
    }

    protected function buildNetwork(object $la, int $numStates, int $numActions)
    {
        $table = new QTable($la,$numStates, $numActions);
        return $table;
    }

    protected function buildPolicy($la)
    {
        $policy = new AnnealingEpsGreedy($la);
        return $policy;
    }

    public function initialize() : void
    {
        //$la = $this->la;
        //$this->q = $la->randomUniform($this->policy->shape(),0,1);
        //$la->multiply($this->policy,$this->q);
        $this->policy->initialize();
    }

    public function resetData() : void
    {
        $this->valueTable->initialize();
    }

    public function isStepUpdate() : bool
    {
        return true;
    }

    public function subStepLength() : int
    {
        return 2;
    }

    protected function estimator() : Estimator
    {
        return $this->valueTable;
    }

    protected function getHistory(ReplayBuffer $experience) : array
    {
        return $experience->recently(1);
    }

    /**
     * $q: (batches)
     * $nextValues: (batches,numActions)
     * $td: 
     */
    protected function tdError(
        NDArray $q,
        NDArray $nextValues,
        float $reward,
        ?array $info,
        iterable $history
        ) : NDArray
    {
        $la = $this->la;
        //  TD = R(t+1)+gamma*max(Q(s(t+1),?))-Q(s(t),a(t))
        if($info!=null) {
            $masks = $this->extractMasks([$info]);
            //$nextValues = $la->nan2num($la->copy($nextValues),alpha:-INF);
            $nextValues = $la->masking($masks,$la->copy($nextValues),fill:-INF);
        }
        $maxQ = $la->reduceMax($nextValues,axis:-1);
        $td = $la->axpy($q,$la->increment(
            $la->scal($this->gamma,$la->copy($maxQ)),$reward),
            -1.0);

        return $td;
    }

    /**
    * @param Any $params
    * @return Any $action
    */
    public function update(ReplayBuffer $experience) : float
    {
        $la = $this->la;

        $history = $this->getHistory($experience);
        [$state,$action,$nextState,$reward,$done,$truncated,$info] = $history[0];
        if($state->shape()!==[1]) {
            throw new LogicException("Shape of State in replay buffer must be (1).".$la->shapeToString($state->shape()));
        }
        $table = $this->valueTable->table(); // table = (numStates,numActions)
        $stateNumber = $la->scalar($la->squeeze($state));
        $actionNumber = $la->scalar($la->squeeze($action));
        $qLocation = $table[$stateNumber][R($actionNumber,$actionNumber+1)];
        // update Q(s(t),a(t))
        if($done) {
            // if done ($nextAction belongs to the next episode)
            // Q(s(t),a(t)) =
            //    Q(s(t),a(t))+eta*(*R(t+1)-Q(s(t),a(t))
            $tdError = $la->increment($la->copy($qLocation),$reward,-1.0);
            $la->axpy($tdError,$qLocation,alpha:$this->eta); // $q += $eta * $tdError
            $error = $la->nrm2($tdError);            // $error = ||$tdError||
        } else {
            // if continued
            // Q(s(t),a(t)) =
            //    Q(s(t),a(t))+eta*TDERROR
            // TDERROR = R(t+1) + gamma*Q(s(t+1),a(t+1)) - Q(s(t),a(t))
            //$nextValues = $table[$nextState];
            $nextValues = $la->gatherb($table,$nextState);
            $tdError = $this->tdError($qLocation,$nextValues,$reward,$info,$history);
            $la->axpy($tdError,$qLocation,$this->eta);
            $error = $la->nrm2($tdError);
        }
        return $error;
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
