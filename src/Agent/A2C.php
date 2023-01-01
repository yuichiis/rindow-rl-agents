<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\A2C\ActorCriticNetwork;
use Rindow\RL\Agents\Network\QNetwork;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;

class A2C extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';
    protected $gamma;
    protected $rewardScaleFactor;
    protected $obsSize;
    protected $actionSize;
    protected $ddqn;
    protected $targetUpdatePeriod;
    protected $targetUpdateTimer;
    protected $lossFn;
    protected $lossOpts;
    protected $optimizer;
    protected $optimizerOpts;
    protected $mo;
    protected $gather;
    protected $trainModel;
    protected $targetModel;
    protected $trainModelGraph;
    protected $trainableVariables;
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
        array $obsSize=null, array $actionSize=null,
        array $fcLayers=null,
        float $epsStart=null, float $epsStop=null, float $epsDecayRate=null,
        EventManager $eventManager=null,
        object $mo = null
        )
    {
        if($network===null) {
            $network = $this->buildNetwork($la,$nn,$obsSize,$actionSize,$fcLayers);
        }
        if(!($network instanceof QPolicy)) {
            echo get_class($network);
            throw new InvalidArgumentException('Network must have Network and QPolicy interfaces.');
        }
        if($policy===null) {
            $policy = $this->buildPolicy($la,$network,$epsStart,$epsStop,$epsDecayRate);
        }
        parent::__construct($la,$policy,$eventManager);

        if($obsSize===null) {
            $obsSize = $network->obsSize();
        }
        if($actionSize===null) {
            $actionSize = $network->actionSize();
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
        $this->obsSize = $obsSize;
        $this->actionSize = $actionSize;
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
            $nn = $network->builder();
        }
        $this->nn = $nn;
        $this->g = $nn->gradient();
        $this->trainModel = $this->buildTrainingModel($la,$nn,$network);
        $this->targetModel = clone $this->trainModel;
        $this->trainModelGraph = $nn->gradient->function([$this->trainModel,'forward']);
        $this->trainableVariables = $this->trainModel->trainableVariables();
        $this->initialize();
    }

    protected function buildNetwork($la,$nn,$obsSize,$actionSize,$fcLayers)
    {
        if($nn===null) {
            throw new InvalidArgumentException('nn must be specifed.');
        }
        if($obsSize===null) {
            throw new InvalidArgumentException('obsSize must be specifed.');
        }
        if($actionSize===null) {
            throw new InvalidArgumentException('actionSize must be specifed.');
        }
        $network = new ActorCriticNetwork($la,$nn,
            $obsSize, $actionSize,fcLayers:$fcLayers);
        return $network;
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
        if($this->gather===null) {
            $this->gather = $nn->layers->Gather(axis:-1);
        }
        $network->compile(loss:$this->lossFn,optimizer:$this->optimizer);
        $network->build(array_merge([1],$this->obsSize),true);

        return $network;
    }
    
    public function summary()
    {
        $this->trainModel->qmodel()->summary();
    }

    protected function buildPolicy($la,$network,$start,$stop,$decayRate)
    {
        $policy = new AnnealingEpsGreedy(
            $la,
            start:$start,stop:$stop,decayRate:$decayRate);
        return $policy;
    }

    public function fileExists(string $filename) : bool
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        return file_exists($filename);
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $this->trainModel->saveWeightsToFile($filename);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $this->trainModel->loadWeightsFromFile($filename);
        $this->targetModel = clone $this->trainModel;
        //$this->trainModelGraph = $nn->gradient->function([$this->trainModel,'forward']);
        //$this->trainableVariables = $this->trainModel->trainableVariables();
        //$this->initialize();
    }

    public function syncWeights($tau=null)
    {
        $this->targetModel->copyWeights($this->trainModel,$tau);
    }

    public function initialize() // : Operation
    {
        //$this->trainModel->compileQModel(
        //    $this->lossFn, $this->lossOpts, $this->optimizer, $this->optimizerOpts);
        //$this->targetModel->compileQModel(
        //    $this->lossFn, $this->lossOpts, $this->optimizer, $this->optimizerOpts);
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

    public function action($observation,bool $training)
    {
        $observation = $this->atleast2d($observation);
        $action = $this->policy->action($this->trainModel,$observation,$training);
        return $action;
    }

    public function getQValue($observation) : float
    {
        if(is_numeric($observation)) {
            $observation = $this->la->array([$observation]);
        } elseif(!($observation instanceof NDArray)) {
            throw new InvalidArgumentException('Observation must be NDArray');
        }
        $qValues = $this->trainModel->getQValues($observation);
        $q = $this->la->max($qValues);
        return $q;
    }

    protected function updateTarget($endEpisode)
    {
        if($this->targetUpdatePeriod > 0) {
            $this->targetUpdateTimer--;
            if($this->targetUpdateTimer <= 0) {
                $this->syncWeights($this->targetUpdateTau);
                $this->targetUpdateTimer = $this->targetUpdatePeriod;
            }
        } else {
            if($endEpisode) {
                $this->syncWeights($this->targetUpdateTau);
            }
        }
    }

    public function update($experience) : float
    {
        $la = $this->la;
        $nn = $this->nn;
        $g  = $this->g;
        $batchSize = $this->batchSize;
        $obsSize = $this->obsSize;

        $transition = $experience->last();
        [$observation,$action,$nextObs,$reward,$endEpisode,$info] = $transition;  // done

        if(!$endEpisode && $experience->size()<$batchSize) {
            return 0.0;
        }

        $states = $la->alloc(array_merge([$batchSize], $obsSize));
        $nextStates = $la->alloc(array_merge([$batchSize], $obsSize));
        $rewards = $la->zeros($la->alloc([$batchSize]));
        $discounts = $la->zeros($la->alloc([$batchSize]));
        $actions = $la->zeros($la->alloc([$batchSize],NDArray::int32));


        if($endEpisode) {
            $discounted = 0;
        } else {
            if(is_scalar($nextObs)) {
                $nextObs = $la->array([[$nextObs]]);
            } elseif($nextObs instanceof NDArray) {
                $nextObs = $la->expandDims($nextObs);
            }
            [$tmp,$v] = $this->trainModel->forward($nextObs,false);
            $discounted = $la->scalar(($la->squeeze($v)));
            unset($tmp);
            unset($v);
        }
        $history = $experience->recently($experience->size());
        $totalReward = 0;
        $i = $steps-1;
        $history = array_reverse($history);
        foreach ($history as $transition) {
            [$observation,$action,$nextObs,$reward,$done,$info] = $transition;
            if($done) {
                $discounted = 0;
            }
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

        // baseline
        $baseline = $K->mean($discountedRewards);
        $la->increment($discountedRewards, -1.0*$baseline);

        $discountedRewards = $g->Variable($discountedRewards);

        // gradients
        $trainModel = $this->model;
        $gather = $this->gather;
        $training = $g->Variable(true);
        $loss = $nn->with($tape=$g->GradientTape(),function() 
                use ($g,$gather,$trainModel,$states,$training,$actions,$discountedRewards) {
            [$actionProbs, $values] = $trainModel($states,$training);
            // action probs
            $actionProbs = $gather->forward([$actionProbs,$actions],$training);
            // advantage
            $advantage = $discountedRewards - $g->stopGradient($values);
            $actionProbs = $g->clipByValue($actionProbs, 1e-10, 1.0);
            $policyLoss = $g->mul($g->log($actionProbs),$advantage);

            // Value loss
            // Mean Squared Error
            $valueLoss = $g->reduceMean($g->pow($g->sub($discounted_rewards,$values) ,2.0), axis:1);

            // policy entropy
            $entropy = $g->reduceSum($g->mul($g->log($actionProbs),$actionProbs), axis:1);

            // total loss
            $value_loss_weight = $g->Variable(0.5);
            $entropy_weight = $g->Variable(0.1);
            $loss = $g->sub( 
                $g->sub(
                    $g->mul($value_loss_weight, $valueLoss),
                    $g->mul($entropy_weight, $entropy)),
                $policyLoss);

            $loss = $g->reduceMean($loss);

            return $loss;
        });
        $grads = $tape->gradient($loss,$this->trainableVariables);
        $this->optimizer->update($this->trainableVariables,$grads);

        $loss = $K->scalar($loss);
        //echo "loss=".$loss."\n";
        return $loss;
    }

}
