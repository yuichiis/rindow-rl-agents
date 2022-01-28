<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
//use Rindow\RL\Agents\Network\DiscreteTrainingModel
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Dqn extends AbstractAgent
{
    protected $la;
    protected $policy;
    protected $gamma;
    protected $rewardScaleFactor;
    protected $obsSize;
    protected $numActions;
    protected $ddqn;
    protected $targetUpdatePeriod;
    protected $targetUpdateTimer;
    protected $lossFn;
    protected $lossOpts;
    protected $optimizer;
    protected $optimizerOpts;
    protected $mo;
    protected $network;
    protected $trainModel;
    protected $targetModel;
    protected $enabledShapeInspection = true;

    public function __construct(
        object $la,
        Network $network=null,
        Policy $policy=null,
        int $batchSize=null,
        float $gamma=null,
        int $targetUpdatePeriod=null,
        float $targetUpdateTau=null,
        bool $ddqn=null,
        object $nn=null,
        object $lossFn=null,
        array $lossOpts=null,
        object $optimizer=null,
        array $optimizerOpts=null,
        array $obsSize=null, int $numActions=null,
        array $fcLayers=null,
        int $epsStart=null, int $epsStop=null, int $epsDecayRate=null,
        object $mo = null
        )
    {
        $this->la = $la;
        if($network===null) {
            $network = $this->buildNetwork($nn,$obsSize,$numActions,$fcLayers);
        }
        if($policy===null) {
            $policy = $this->buildPolicy($network,$epsStart,$epsStop,$epsDecayRate);
        }
        if($obsSize===null) {
            $obsSize = $network->obsSize();
        }
        if($numActions===null) {
            $numActions = $network->numActions();
        }
        if($batchSize===null) {
            $batchSize = 32;
        }
        if($gamma===null) {
            $gamma = 0.99;
        }
        if($targetUpdatePeriod===null) {
            $targetUpdatePeriod = 100;
        }
        if($targetUpdateTau===null) {
            $targetUpdateTau = 1.0;
        }
        if($ddqn===null) {
            $ddqn = false;
        }
        $this->network = $network;
        $this->policy = $policy;
        $this->obsSize = $obsSize;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->gamma = $gamma;
        $this->targetUpdatePeriod = $targetUpdatePeriod;
        $this->targetUpdateTau = $targetUpdateTau;
        $this->ddqn = $ddqn;
        $this->lossFn = $lossFn;
        $this->lossOpts = $lossOpts;
        $this->optimizer = $optimizer;
        $this->optimizerOpts = $optimizerOpts;
        $this->mo = $mo;
        if($nn===null) {
            $nn = $this->network->builder();
        }
        $this->trainModel = $this->buildTrainingModel($la,$nn,$this->network,);
        $this->targetModel = clone $this->trainModel;
        $this->initialize();
    }

    protected function buildNetwork($nn,$obsSize,$numActions,$fcLayers)
    {
        if($nn===null) {
            throw new InvalidArgumentException('nn must be specifed.');
        }
        if($obsSize===null) {
            throw new InvalidArgumentException('obsSize must be specifed.');
        }
        if($numActions===null) {
            throw new InvalidArgumentException('numActions must be specifed.');
        }
        $network = new QNetwork(
            $nn, $obsSize, $numActions,$fcLayers);
        return $network;
    }

    protected function buildTrainingModel($la,$nn,$network)
    {
        $model = new DiscreteTrainingModel($la,$nn,$network);
        return $model;
    }

    public function summary()
    {
        $this->network->qmodel()->summary();
    }

    protected function buildPolicy($network,$start,$stop,$decayRate)
    {
        $policy = new AnnealingEpsGreedy(
            $this->la, $network,
            $start,$stop,$decayRate);
        return $policy;
    }

    public function syncWeights($tau=null)
    {
        $this->targetModel->copyWeights($this->trainModel,$tau);
    }

    public function initialize() // : Operation
    {
        $this->trainModel->compileQModel(
            $this->lossFn, $this->lossOpts, $this->optimizer, $this->optimizerOpts);
        $this->targetModel->compileQModel(
            $this->lossFn, $this->lossOpts, $this->optimizer, $this->optimizerOpts);
        if($this->targetUpdatePeriod>0) {
            $this->targetUpdateTimer = $this->targetUpdatePeriod;
        }
        $this->syncWeights($tau=1.0);
    }

    public function isStepUpdate() : bool
    {
        return true;
    }

    public function subStepLength() : int
    {
        return 1;
    }

    public function startEpisode(int $episode) : void
    {}

    public function endEpisode(int $episode) : void
    {
        if($this->targetUpdatePeriod <= 0) {
            $this->syncWeights($this->targetUpdateTau);
        }
    }

    public function action($observation,bool $training)
    {
        //if(is_numeric($observation)) {
        //    $observation = $this->la->array([$observation]);
        //} elseif(!($observation instanceof NDArray)) {
        //    throw new InvalidArgumentException('Observation must be NDArray');
        //}
        if($training) {
            $action = $this->policy->action($observation,$this->elapsedTime);
        } else {
            $values = $this->network->getQValues($observation);
            $action = $this->la->imax($values);
        }
        return $action;
    }

    public function getQValue($observation)
    {
        if(is_numeric($observation)) {
            $observation = $this->la->array([$observation]);
        } elseif(!($observation instanceof NDArray)) {
            throw new InvalidArgumentException('Observation must be NDArray');
        }
        $qValues = $this->network->getQValues($observation);
        $q = $this->la->max($qValues);
        return $q;
    }

    public function update($experience)
    {
        $la = $this->la;
        $batchSize = $this->batchSize;
        $obsSize = $this->obsSize;
        $numActions = $this->numActions;

        if($experience->size()<$batchSize) {
            return;
        }
        $states = $la->alloc(array_merge([$batchSize], $obsSize));
        $nextStates = $la->alloc(array_merge([$batchSize], $obsSize));
        $rewards = $la->zeros($la->alloc([$batchSize]));
        $discounts = $la->zeros($la->alloc([$batchSize]));
        $actions = $la->zeros($la->alloc([$batchSize],NDArray::int32));

        $batch = $experience->sample($batchSize);
        $i = 0;
        foreach($batch as $transition) {
            [$observation,$action,$nextObs,$reward,$done,$info] = $transition;
            if(is_numeric($observation)) {
                $states[$i][0] = $observation;
            } else {
                $la->copy($observation,$states[$i]);
            }
            if(is_numeric($nextObs)) {
                $nextStates[$i][0] = $observation;
            } else {
                $la->copy($nextObs,$nextStates[$i]);
            }
            $rewards[$i] = $reward;
            $discounts[$i] = $done ? 0.0 : 1.0;
            $actions[$i] = $action;
            $i++;
        }
        $nextQValues = $this->targetModel->qNetwork()->getQValuesBatch(
                $nextStates);
        if($this->ddqn) {
            $nextActions = $this->network->getQValuesBatch($nextStates);
            $nextActions = $la->reduceArgMax($nextActions,$axis=-1);
            $nextQValues = $la->gather($nextQValues,$nextActions,$axis=-1);
        } else {
            $nextQValues = $la->reduceMax($nextQValues,$axis=-1);
        }
        $la->scal($this->gamma,$nextQValues);
        $la->multiply($discounts,$nextQValues);
        $la->axpy($rewards,$nextQValues);

        $history = $this->trainModel->fit([$states,$actions],$nextQValues,
            ['batch_size'=>$batchSize,'epochs'=>1, 'verbose'=>0]);

        if($this->enabledShapeInspection) {
            $this->trainModel->setShapeInspection(false);
            $this->targetModel->setShapeInspection(false);
            $this->enabledShapeInspection = false;
        }
        if($this->targetUpdatePeriod > 0) {
            $this->targetUpdateTimer--;
            if($this->targetUpdateTimer <= 0) {
                $this->syncWeights($this->targetUpdateTau);
                $this->targetUpdateTimer = $this->targetUpdatePeriod;
            }
        }

        return $history['loss'][0];
    }
}
