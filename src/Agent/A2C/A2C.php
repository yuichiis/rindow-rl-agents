<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Gradient\GraphFunction;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\AbstractAgent;
use Rindow\RL\Agents\Policy\Boltzmann;
use function Rindow\Math\Matrix\R;

class A2C extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';
    protected float $gamma;
    //protected $rewardScaleFactor;
    protected array $stateShape;
    protected int $numActions;
    protected int $targetUpdatePeriod;
    protected int $targetUpdateTimer;
    protected Optimizer $optimizer;
    protected array $optimizerOpts;
    protected ?object $mo;
    protected Builder $nn;
    protected object $g;
    protected Layer $gather;
    protected ActorCriticNetwork $trainModel;
    protected ActorCriticNetwork $targetModel;
    protected GraphFunction $trainModelGraph;
    protected $trainableVariables;
    protected bool $enabledShapeInspection = true;
    protected int $targetUpdateTau;
    protected int $batchSize;

    public function __construct(
        object $la,
        ?Network $network=null,
        ?Policy $policy=null,
        ?int $batchSize=null,
        ?float $gamma=null,
        ?int $targetUpdatePeriod=null,
        ?float $targetUpdateTau=null,
        ?object $nn=null,
        ?object $optimizer=null,
        ?array $optimizerOpts=null,
        ?array $stateShape=null, ?int $numActions=null,
        ?array $fcLayers=null,
        ?float $policyTau=null,?float $policyMin=null,?float $policyMax=null,
        ?EventManager $eventManager=null,
        ?object $mo = null
        )
    {
        $network ??= $this->buildNetwork($la,$nn,$stateShape,$numActions,$fcLayers);
        if(!($network instanceof Estimator)) {
            echo get_class($network);
            throw new InvalidArgumentException('Network must have Network and Estimator interfaces.');
        }
        $policy ??= $this->buildPolicy($la,$policyTau,$policyMin,$policyMax);
        parent::__construct($la,$policy,$eventManager);

        $stateShape ??= $network->stateShape();
        $numActions ??= $network->numActions();
        $batchSize ??= 32;
        $gamma ??= 0.99;
        $targetUpdatePeriod ??= 100;
        $targetUpdateTau ??= 1.0;
        $nn ??= $network->builder();
        $optimizerOpts ??= [];
        $optimizer = $nn->optimizers->Adam(...$optimizerOpts);

        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->gamma = $gamma;
        $this->targetUpdatePeriod = $targetUpdatePeriod;
        $this->targetUpdateTau = $targetUpdateTau;
        $this->optimizer = $optimizer;
        $this->optimizerOpts = $optimizerOpts;
        $this->mo = $mo;
        $this->nn = $nn;
        $this->g = $nn->gradient();
        $this->trainModel = $this->buildTrainingModel($la,$nn,$network);
        $this->targetModel = clone $this->trainModel;
        $this->trainModelGraph = $nn->gradient->function([$this->trainModel,'forward']);
        $this->trainableVariables = $this->trainModel->trainableVariables();
        $this->initialize();
    }

    protected function buildNetwork($la,$nn,$stateShape,$numActions,$fcLayers)
    {
        if($nn===null) {
            throw new InvalidArgumentException('nn must be specifed.');
        }
        if($stateShape===null) {
            throw new InvalidArgumentException('stateShape must be specifed.');
        }
        if($numActions===null) {
            throw new InvalidArgumentException('numActions must be specifed.');
        }
        $network = new ActorCriticNetwork($la,$nn,
            $stateShape, $numActions,fcLayers:$fcLayers);
        return $network;
    }

    protected function buildTrainingModel(
        $la,$nn,$network,
        )
    {
        $this->gather = $nn->layers->Gather(axis:-1);
        $network->build(array_merge([1],$this->stateShape));

        return $network;
    }
    
    public function summary()
    {
        $this->trainModel->summary();
    }

    protected function buildPolicy(
        object $la,
        ?float $tau=null,
        ?float $min=null,
        ?float $max=null,
        )
    {
        $fromLogits = true;
        $policy = new Boltzmann(
            $la,
            $tau,
            $min,
            $max,
            $fromLogits,
        );
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

    public function initialize() : void // : Operation
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

    protected function estimator() : Estimator
    {
        return $this->trainModel;
    }

    public function getQValue($state) : float
    {
        if(is_numeric($state)) {
            $state = $this->la->array([$state]);
        } elseif(!($state instanceof NDArray)) {
            throw new InvalidArgumentException('state must be NDArray');
        }
        $qValues = $this->trainModel->getActionValues($state);
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
        $K  = $nn->backend();
        $batchSize = $this->batchSize;
        $stateShape = $this->stateShape;

        $transition = $experience->last();
        [$state,$action,$nextState,$reward,$terminated,$truncated,$info] = $transition;  // done

        if(!$terminated && $experience->size()<$batchSize) {
            return 0.0;
        }

        $states = $la->alloc(array_merge([$batchSize], $stateShape));
        $nextStates = $la->alloc(array_merge([$batchSize], $stateShape));
        $rewards = $la->zeros($la->alloc([$batchSize]));
        $discounts = $la->zeros($la->alloc([$batchSize]));
        $actions = $la->zeros($la->alloc([$batchSize],NDArray::int32));
        $discountedRewards = $la->zeros($la->alloc([$batchSize,1]));;


        if($terminated) {
            $discounted = 0;
        } else {
            if(!($nextState instanceof NDArray)) {
                throw new LogicException("nextState must be NDArray");
            }
            if($la->isInt($state)) {
                $state = $la->astype($state,dtype:NDArray::float32);
            }
            if($nextState->ndim()) {
                $nextState = $la->expandDims($nextState,axis:0);
            }
            [$tmp,$v] = $this->trainModel->forward($nextState,false);
            $discounted = $la->scalar(($la->squeeze($v)));
            unset($tmp);
            unset($v);
        }
        $history = $experience->recently($experience->size());
        $totalReward = 0;
        $i = $batchSize-1;
        $history = array_reverse($history);
        foreach ($history as $transition) {
            [$state,$action,$nextState,$reward,$done,$truncated,$info] = $transition;
            if($done) {
                $discounted = 0;
            }
            $discounted = $reward + $discounted*$this->gamma;
            $discountedRewards[$i][0] = $discounted;
            $rewards[$i] = $reward;
            if(!($state instanceof NDArray)) {
                throw new LogicException("state must be NDArray.");
            }
            if($la->isInt($state)) {
                $state = $la->astype($state,dtype:NDArray::float32);
            }
            $la->copy($state,$states[$i]);
            if(!($action instanceof NDArray)) {
                throw new LogicException("action must be NDArray.");
            }
            if($action->ndim()!==0) {
                throw new LogicException("shape of action must be scalar ndarray.");
            }
            $la->copy($action->reshape([1]),$actions[R($i,$i+1)]);
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
        $trainModel = $this->trainModel;
        $gather = $this->gather;
        $training = $g->Variable(true);
        $loss = $nn->with($tape=$g->GradientTape(),function() 
                use ($g,$gather,$trainModel,$states,$training,$actions,$discountedRewards) {
            [$actionProbs, $values] = $trainModel($states,$training);
            // action probs
            $actionProbs = $gather->forward([$actionProbs,$actions],$training);
            // advantage
            $advantage = $g->sub($discountedRewards,$g->stopGradient($values));
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
