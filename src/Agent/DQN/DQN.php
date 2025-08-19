<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\AbstractAgent;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Gradient\GraphFunction;
use function Rindow\Math\Matrix\R;

class DQN extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';
    protected float $gamma;
    protected array $stateShape;
    protected int $numActions;
    protected bool $ddqn;
    protected int $targetUpdatePeriod;
    protected int $targetUpdateTimer;
    protected Loss $lossFn;
    protected Optimizer $optimizer;
    protected ?object $mo;
    protected QNetwork $trainModel;
    protected QNetwork $targetModel;
    protected GraphFunction $trainModelGraph;
    protected array $trainableVariables;
    protected bool $enabledShapeInspection = true;
    protected Builder $nn;
    protected object $g;
    protected int $batchSize;
    protected float $targetUpdateTau;

    public function __construct(
        object $la,
        ?Network $network=null,
        ?Policy $policy=null,
        ?int $batchSize=null,
        ?float $gamma=null,
        ?int $targetUpdatePeriod=null,
        ?float $targetUpdateTau=null,
        ?bool $ddqn=null,
        ?object $nn=null,
        ?object $lossFn=null,
        ?array $lossOpts=null,
        ?object $optimizer=null,
        ?array $optimizerOpts=null,
        ?array $stateShape=null, ?int $numActions=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ?float $epsStart=null, ?float $epsStop=null, ?float $epsDecayRate=null,
        ?bool $episodeAnnealing=null,
        ?string $stateField=null,
        ?object $mo = null
        )
    {
        $network ??= $this->buildNetwork($la,$nn,$stateShape,$numActions,$fcLayers,$activation,$kernelInitializer);
        if(!($network instanceof Estimator)) {
            throw new InvalidArgumentException('Network must have Network and Estimator interfaces.');
        }
        $policy ??= $this->buildPolicy($la,$epsStart,$epsStop,$epsDecayRate,$episodeAnnealing);
        parent::__construct($la,policy:$policy,stateField:$stateField);

        $stateShape ??= $network->stateShape();
        $numActions ??= $network->numActions();
        $batchSize ??= 32;
        $gamma ??= 0.99;
        $targetUpdatePeriod ??= 100;
        $targetUpdateTau ??= 1.0;
        $ddqn ??= false;

        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->gamma = $gamma;
        $this->targetUpdatePeriod = $targetUpdatePeriod;
        $this->targetUpdateTau = $targetUpdateTau;
        $this->ddqn = $ddqn;
        $this->mo = $mo;
        if($nn===null) {
            $nn = $network->builder();
        }
        $this->nn = $nn;
        $this->g = $nn->gradient();
        $this->trainModel = $this->buildTrainingModel(
            $la,$nn,$network,$lossFn,$lossOpts,$optimizer,$optimizerOpts
        );
        $this->targetModel = clone $this->trainModel;
        $this->trainModelGraph = $nn->gradient->function([$this->trainModel,'forward']);
        $this->trainableVariables = $this->trainModel->trainableVariables();
        $this->initialize();
    }

    protected function buildNetwork(
        object $la, Builder $nn,
        array $stateShape, int $numActions,
        array $fcLayers,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ) : Network
    {
        //if($nn===null) {
        //    throw new InvalidArgumentException('nn must be specifed.');
        //}
        //if($stateShape===null) {
        //    throw new InvalidArgumentException('stateShape must be specifed.');
        //}
        //if($numActions===null) {
        //    throw new InvalidArgumentException('numActions must be specifed.');
        //}
        $network = new QNetwork(
            $la,$nn,
            $stateShape, $numActions,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
        );
        return $network;
    }

    protected function buildTrainingModel(
        object $la, Builder $nn, Network $network,
        ?Loss $lossFn, ?array $lossOpts,
        ?Optimizer $optimizer, ?array $optimizerOpts,
        ) : Network
    {
        $lossOpts ??= [];
        $optimizerOpts ??= [];
        $lossFn ??= $nn->losses->Huber(...$lossOpts);
        $optimizer ??= $nn->optimizers->Adam(...$optimizerOpts);

        $this->lossFn = $lossFn;
        $this->optimizer = $optimizer;

        $network->compile(loss:$this->lossFn,optimizer:$this->optimizer);
        $network->build(array_merge([1],$this->stateShape));

        return $network;
    }
    
    public function summary() : void
    {
        $this->trainModel->model()->summary();
    }

    protected function buildPolicy(
        object $la,
        ?float $start,
        ?float $stop,
        ?float $decayRate,
        ?bool $episodeAnnealing,
        ) : Policy
    {
        $policy = new AnnealingEpsGreedy(
            $la,
            start:$start,stop:$stop,decayRate:$decayRate,episodeAnnealing:$episodeAnnealing);
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

    public function syncWeights(?float $tau=null)
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
        return $this->batchSize;
    }

    public function numRolloutSteps() : int
    {
        return 1;
    }

    protected function estimator() : Estimator
    {
        return $this->trainModel;
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
        $stateShape = $this->stateShape;
        $numActions = $this->numActions;

        if($experience->size()<$batchSize) {
            return 0.0;
        }
        $transition = $experience->last();
        $endEpisode = $transition[4];  // done

        $states = $la->alloc(array_merge([$batchSize], $stateShape),dtype:NDArray::float32);
        $nextStates = $la->alloc(array_merge([$batchSize], $stateShape),dtype:NDArray::float32);
        $rewards = $la->zeros($la->alloc([$batchSize],dtype:NDArray::float32));
        $discounts = $la->zeros($la->alloc([$batchSize],dtype:NDArray::float32));
        $actions = $la->zeros($la->alloc([$batchSize],NDArray::int32));
        $masks = $la->ones($la->alloc([$batchSize, $numActions],NDArray::bool));

        $batch = $experience->sample($batchSize);
        $i = 0;
        foreach($batch as $transition) {
            [$obs,$action,$nextObs,$reward,$done,$truncated,$info] = $transition;
            $state = $this->extractState($obs);
            if(!($state instanceof NDArray)) {
                throw new LogicException("state must be NDArray.");
            }
            if($la->isInt($state)) {
                $state = $la->astype($state,dtype:NDArray::float32);
            }
            $la->copy($state,$states[$i]);
            $nextState = $this->extractState($nextObs);
            if(!($nextState instanceof NDArray)) {
                throw new LogicException("nextState must be NDArray.");
            }
            if($la->isInt($nextState)) {
                $nextState = $la->astype($nextState,dtype:NDArray::float32);
            }
            $la->copy($nextState,$nextStates[$i]);
            $rewards[$i] = $reward;
            $discounts[$i] = $done ? 0.0 : 1.0;
            if(!($action instanceof NDArray)) {
                throw new LogicException("action must be NDArray.");
            }
            if($action->ndim()!==0) {
                throw new LogicException("shape of action must be scalar ndarray.");
            }
            $la->copy($action->reshape([1]),$actions[R($i,$i+1)]);
            $mask = $this->extractMask($nextObs);
            if($mask!==null) {
                $la->copy($mask,$masks[$i]);
            }
            $i++;
        }

        //
        // calculate netQValues from the target model
        //
        $nextQValues = $this->targetModel->getActionValues($nextStates);
        $la->masking($masks,$nextQValues,fill:-INF);
        //echo $this->mo->toString($nextQValues,format:'%5.3f',indent:true)."\n";
        if($this->ddqn) {
            $nextActions = $this->trainModel->getActionValues($nextStates);
            $la->masking($masks,$nextActions,fill:-INF);
            $nextActions = $la->reduceArgMax($nextActions,axis:-1,dtype:NDArray::int32);
            $mo = $this->trainModel->backend()->localMatrixOperator();
            //echo "NEXTQVALUES:".$mo->shapeToString($nextQValues->shape())."\n";
            //echo "nextActions:".$mo->shapeToString($nextActions->shape())."\n";
            $nextQValues = $la->gatherb($nextQValues,$nextActions,batchDims:-1);
            //echo "nextQValues:".$mo->shapeToString($nextQValues->shape())."\n";
            //$nextQValues = $la->gather($nextQValues,$nextActions,axis:-1,indexDepth:-1);
        } else {
            $nextQValues = $la->reduceMax($nextQValues,axis:-1);
        }
        //echo $this->mo->toString($nextQValues,format:'%5.3f',indent:true)."\n";
        $la->scal($this->gamma,$nextQValues);
        $la->multiply($discounts,$nextQValues);
        $la->axpy($rewards,$nextQValues);


        //
        // training the traing model
        //
        //$history = $this->trainModel->fit([$states,$actions],$nextQValues,
        //    batch_size:$batchSize,epochs:1, verbose:0);
        $trainModel = $this->trainModelGraph;
        $lossFn = $this->lossFn;

        $states = $g->Variable($states);
        $actions = $g->Variable($actions);
        $nextQValues = $g->Variable($nextQValues);
        $training = $g->Variable(true);
        $loss = $nn->with($tape=$g->GradientTape(), function()
                use ($g,$trainModel,$lossFn,$states,$actions,$nextQValues,$training) {
            $qValues = $trainModel($states,$training);
            $qValues = $g->gather($qValues,$actions,batchDims:-1);
            $loss = $lossFn->forward($nextQValues,$qValues);
            return $loss;
        });
        $grads = $tape->gradient($loss,$this->trainableVariables);
        $this->optimizer->update($this->trainableVariables,$grads);

        if($this->enabledShapeInspection) {
            $this->trainModel->setShapeInspection(false);
            $this->targetModel->setShapeInspection(false);
            $this->enabledShapeInspection = false;
        }

        //
        // update the target model from the training model
        //
        $this->updateTarget($endEpisode);

        $loss = $nn->backend()->scalar($loss->value());
        if($this->metrics->isAttracted('loss')) {
            $this->metrics->update('loss',$loss);
        }
        return $loss;
    }
}
