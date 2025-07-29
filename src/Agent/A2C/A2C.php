<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Loss\Loss;
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
    protected float $valueLossWeight;
    protected float $entropyWeight;
    //protected $rewardScaleFactor;
    protected array $stateShape;
    protected int $numActions;
    protected Loss $lossFunc;
    protected Optimizer $optimizer;
    protected array $optimizerOpts;
    protected ?object $mo;
    protected Builder $nn;
    protected object $g;
    protected ActorCriticNetwork $model;
    protected GraphFunction $modelGraph;
    protected $trainableVariables;
    protected bool $enabledShapeInspection = true;
    protected int $batchSize;

    public function __construct(
        object $la,
        ?Network $network=null,
        ?Policy $policy=null,
        ?int $batchSize=null,
        ?float $gamma=null,
        ?float $valueLossWeight=null,
        ?float $entropyWeight=null,
        ?object $nn=null,
        ?object $optimizer=null,
        ?array $optimizerOpts=null,
        ?Loss $lossFunc=null,
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
        $valueLossWeight ??= 0.5;
        $entropyWeight ??= 0.0;
        $nn ??= $network->builder();
        $optimizerOpts ??= ['lr'=>7e-4];
        $optimizer ??= $nn->optimizers->Adam(...$optimizerOpts);
        //$optimizer ??= $nn->optimizers->RMSprop(...$optimizerOpts);
        //$lossFunc ??= $nn->losses()->Huber();
        $lossFunc ??= $nn->losses()->MeanSquaredError();

        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->gamma = $gamma;
        $this->valueLossWeight = $valueLossWeight;
        $this->entropyWeight = $entropyWeight;
        $this->optimizer = $optimizer;
        $this->optimizerOpts = $optimizerOpts;
        $this->lossFunc = $lossFunc;
        $this->mo = $mo;
        $this->nn = $nn;
        $this->g = $nn->gradient();
        $this->model = $this->buildModel($la,$nn,$network);
        $this->modelGraph = $nn->gradient->function([$this->model,'forward']);
        $this->trainableVariables = $this->model->trainableVariables();
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

    protected function buildModel(
        $la,$nn,$network,
        )
    {
        $network->build(array_merge([1],$this->stateShape));

        return $network;
    }
    
    public function summary()
    {
        $this->model->summary();
    }

    protected function buildPolicy(
        object $la,
        ?float $tau=null,
        ?float $min=null,
        ?float $max=null,
        )
    {
        $tau ??= 1.0;
        $min ??= -INF;
        $max ??= INF;
        $policy = new Boltzmann(
            $la,
            $tau,
            $min,
            $max,
            fromLogits:true,
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
        $this->model->saveWeightsToFile($filename);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $this->model->loadWeightsFromFile($filename);
        //$this->modelGraph = $nn->gradient->function([$this->model,'forward']);
        //$this->trainableVariables = $this->model->trainableVariables();
        //$this->initialize();
    }

    public function syncWeights($tau=null) : void
    {
    }

    public function initialize() : void // : Operation
    {
        //$this->model->compileQModel(
        //    $this->lossFn, $this->lossOpts, $this->optimizer, $this->optimizerOpts);
        //$this->targetModel->compileQModel(
        //    $this->lossFn, $this->lossOpts, $this->optimizer, $this->optimizerOpts);
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
        return $this->model;
    }

    protected function updateTarget($endEpisode)
    {
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
        [$state,$action,$nextState,$reward,$done,$truncated,$info] = $transition;  // done

        if($experience->size()<$batchSize) {
            return 0.0;
        }

        $states = $la->alloc(array_merge([$batchSize], $stateShape));
        $nextStates = $la->alloc(array_merge([$batchSize], $stateShape));
        $rewards = $la->zeros($la->alloc([$batchSize]));
        $discounts = $la->zeros($la->alloc([$batchSize]));
        $actions = $la->zeros($la->alloc([$batchSize],NDArray::int32));
        $discountedRewards = $la->zeros($la->alloc([$batchSize,1]));;


        if($done) {
            $discounted = 0;
        } else {
            if(!($nextState instanceof NDArray)) {
                throw new LogicException("nextState must be NDArray");
            }
            if($la->isInt($nextState)) {
                $nextState = $la->astype($nextState,dtype:NDArray::float32);
            }
            if($nextState->ndim()) {
                $nextState = $la->expandDims($nextState,axis:0);
            }
            [$tmp,$v] = $this->model->forward($nextState,false);
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
        #onehot_actions = tf.one_hot(actions, model.output[0].shape[1])

        $valueLossWeight = $g->Variable($this->valueLossWeight);
        $entropyWeight = $g->Variable($this->entropyWeight);
        // gradients
        $model = $this->model;
        $lossFunc = $this->lossFunc;
        $training = $g->Variable(true);
        $loss = $nn->with($tape=$g->GradientTape(),function() 
            use ($g,$lossFunc,$model,$states,$training,$actions,$discountedRewards,
                $valueLossWeight,$entropyWeight)
        {
            [$logits, $values] = $model($states,$training);

            // advantage
            $advantage = $g->sub($discountedRewards,$g->stopGradient($values));

            // action probs
            $actionProbs = $g->softmax($logits);
            $selectedActionProbs = $g->expandDims($g->gather($actionProbs,$actions,batchDims:-1),-1);
            $selectedActionProbs = $g->clipByValue($selectedActionProbs, 1e-10, 1.0);

            // policy loss
            $policyLoss = $g->scale(-1,$g->mul($g->log($selectedActionProbs),$advantage));
            
            // Value loss
            // SB3ではMSE固定
            $valueLoss = $lossFunc($discountedRewards,$values);

            // policy entropy
            $actionProbsCliped = $g->clipByValue($actionProbs, 1e-10, 1.0);
            $entropy = $g->scale(-1,$g->reduceSum($g->mul($actionProbsCliped,$g->log($actionProbsCliped)), axis:1, keepdims:true));

            // total loss
            $loss = $g->add(
                $g->add(
                    $g->mul($valueLossWeight, $valueLoss),
                    $g->mul($entropyWeight, $entropy)
                ),
                $policyLoss
            );

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
