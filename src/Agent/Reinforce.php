<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Network\ActorNetwork;
use Rindow\RL\Agents\Policy\Boltzmann;

class Reinforce extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';

    protected $gamma;
    protected $useBaseline;
    protected $obsSize;
    protected $numActions;
    protected $lossFn;
    protected $lossOpts;
    protected $optimizer;
    protected $optimizerOpts;
    protected $mo;
    protected $gather;
    protected $model;
    protected $trainModelGraph;
    protected $trainableVariables;

    public function __construct(
        object $la,
        Network $network=null,
        Policy $policy=null,
        float $gamma=null,
        bool $useBaseline=null,
        object $nn=null,
        object $optimizer=null,
        array $optimizerOpts=null,
        array $obsSize=null, int $numActions=null,
        array $fcLayers=null,
        float $boltzTau=null,
        EventManager $eventManager=null,
        object $mo = null
        )
    {
        if($network===null) {
            $network = $this->buildNetwork($la,$nn,$obsSize,$numActions,$fcLayers);
        }
        if(!($network instanceof QPolicy)) {
            throw new InvalidArgumentException('Network must have Network and QPolicy interfaces.');
        }
        if($policy===null) {
            $policy = $this->buildPolicy($la,$network,$boltzTau);
        }
        parent::__construct($la,$policy,$eventManager);

        if($obsSize===null) {
            $obsSize = $network->obsSize();
        }
        if($numActions===null) {
            $numActions = $network->numActions();
        }
        if($gamma===null) {
            $gamma = 0.99;
        }
        $this->obsSize = $obsSize;
        $this->numActions = $numActions;
        $this->gamma = $gamma;
        $this->useBaseline = $useBaseline;
        $this->optimizer = $optimizer;
        $this->optimizerOpts = $optimizerOpts;
        $this->mo = $mo;

        if($nn===null) {
            $nn = $network->builder();
        }
        $this->nn = $nn;
        $this->g = $nn->gradient();
        $this->model = $this->buildTrainingModel($la,$nn,$network);
        $this->trainModelGraph = $nn->gradient->function([$this->model,'forward']);
        $this->trainableVariables = $this->model->trainableVariables();
        $this->initialize();
    }

    protected function buildNetwork($la,$nn,$obsSize,$numActions,$fcLayers)
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
        $actionSize = [$numActions];
        $network = new ActorNetwork($la,$nn,
            $obsSize, $actionSize,fcLayers:$fcLayers);
        return $network;
    }

    protected function buildPolicy($la,$network,$tau)
    {
        $policy = new Boltzmann(
            $la,
            tau:$tau);
        return $policy;
    }

    protected function buildTrainingModel(
        $la,$nn,$network,
        )
    {
        $lossOpts = $this->lossOpts;
        $optimizerOpts = $this->optimizerOpts;
        if($lossOpts===null) {
            $lossOpts=[];
        }
        if($optimizerOpts===null) {
            $optimizerOpts=[];
        }
        if($this->lossFn===null) {
            $this->lossFn=$nn->losses->Huber(...$lossOpts);
        }
        if($this->optimizer===null) {
            $this->optimizer=$nn->optimizers->Adam(...$optimizerOpts);
        }
        $this->gather = $nn->layers->Gather(axis:-1);
        $network->compile(loss:$this->lossFn,optimizer:$this->optimizer);
        $network->build(array_merge([1],$this->obsSize),true);

        return $network;
    }
    
    public function summary()
    {
        $this->model->summary();
    }

    public function fileExists(string $filename) : bool
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        return file_exists($filename);
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $this->model->saveWeightsToFile($filename);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $this->model->loadWeightsFromFile($filename);
    }

    public function initialize() // : Operation
    {
    }

    public function isStepUpdate() : bool
    {
        return false;
    }

    public function subStepLength() : int
    {
        return 1;
    }

    protected function policyTable() : QPolicy
    {
        return $this->model;
    }

    public function getQValue($observation) : float
    {
        if(is_numeric($observation)) {
            $observation = $this->la->array([$observation]);
        } elseif(!($observation instanceof NDArray)) {
            throw new InvalidArgumentException('Observation must be NDArray');
        }
        $qValues = $this->model->getQValues($observation);
        $q = $this->la->max($qValues);
        return $q;
    }

    /**
    * @param Any $params
    */
    public function update($experience) : float
    {
        $la = $this->la;
        $nn = $this->nn;
        $K = $nn->backend();
        $g  = $this->g;
        $obsSize = $this->obsSize;
        $numActions = $this->numActions;

        $steps = $experience->size();
        $rewards = $la->alloc([$steps]);
        $states = $la->alloc(array_merge([$steps], $obsSize));
        $discountedRewards = $la->alloc([$steps]);
        $actions = $la->alloc([$steps],NDArray::int32);
        $discounts = $la->zeros($la->alloc([$steps]));

        $history = $experience->recently($steps);

        $totalReward = 0;
        $i = $steps-1;
        $discounted = 0;
        $history = array_reverse($history);
        foreach ($history as $transition) {
            [$observation,$action,$nextObs,$reward,$done,$info] = $transition;
            $discounted = $reward + $discounted*$this->gamma;
            $discountedRewards[$i] = $discounted;
            $rewards[$i] = $reward;
            if(is_numeric($observation)) {
                $states[$i][0] = $observation;
            } else {
                $la->copy($observation,$states[$i]);
            }
            $actions[$i] = $action;
            if(!$done) {
                $discounts[$i] = 1.0;
            }
            $i--;
        }
        $experience->clear();

        //echo "===\n";

        //echo "actions=".$this->mo->toString($actions,'%5d')."\n";
        //echo "rewards=".$this->mo->toString($rewards,'%5.5f')."\n";
        //echo "discounts=".$this->mo->toString($discounts,'%5.5f')."\n";


        if($this->useBaseline) {
            //echo "dscReturn=".$this->mo->toString($discountedRewards,'%5.5f')."\n";
            //$la->multiply($discounts,$discountedRewards);
            $baseline = $K->mean($discountedRewards);
            //echo "baseline=".$baseline."\n";
            //$std = $K->std($discountedRewards);
            //$epsilon = $K->epsilon();
            //echo "std=".$std."\n";
            $la->increment($discountedRewards, -1.0*$baseline);
            //echo "AvgReturn=".$this->mo->toString($discountedRewards,'%5.5f')."\n";
            //$la->scal(1/($std+$epsilon),$discountedRewards);
            //echo "stdReturn=".$this->mo->toString($discountedRewards,'%5.5f')."\n";
        }
        $discountedRewards = $g->Variable($discountedRewards);

        $trainModel = $this->model;
        $gather = $this->gather;
        $training = $g->Variable(true);
        $loss = $nn->with($tape=$g->GradientTape(),function() 
                use ($g,$gather,$trainModel,$states,$training,$actions,$discountedRewards) {
            $qValues = $trainModel($states,$training);
            $qValues = $gather->forward([$qValues,$actions],$training);
            $qValues = $g->clipByValue($qValues, 1e-10, 1.0);
            $qValues = $g->log($qValues);
            $loss = $g->mul($qValues,$discountedRewards);
            $loss = $g->mul($g->Variable(-1),$g->reduceSum($loss));
            return $loss;
        });
        $grads = $tape->gradient($loss,$this->trainableVariables);
        $this->optimizer->update($this->trainableVariables,$grads);

        $loss = $K->scalar($loss);
        //echo "loss=".$loss."\n";
        return $loss;
    }
}
