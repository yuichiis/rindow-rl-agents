<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Gradient\GraphFunction;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Policy\OUNoise;
use Rindow\RL\Agents\Agent\AbstractAgent;

use InvalidArgumentException;

class DDPG extends AbstractAgent
{
    const ACTOR_FILENAME = '%s-actor.model';
    const CRITIC_FILENAME = '%s-critic.model';
    protected float $gamma;
    protected array $stateShape;
    protected int $numActions;
    protected int $targetUpdatePeriod;
    protected int $targetUpdateTimer;
    protected ?Optimizer $optimizer;
    protected ?array $optimizerOpts;
    protected ?object $mo;
    protected bool $enabledShapeInspection = true;
    protected ?Builder $nn;
    protected Model $actorModel;
    protected Model $criticModel;
    protected Model $targetActor;
    protected Model $targetCritic;
    protected int $batchSize;
    protected NDArray $lowerBound;
    protected NDArray $upperBound;
    protected float $targetUpdateTau;
    protected Optimizer $criticOptimizer;
    protected Optimizer $actorOptimizer;
    //protected GraphFunction $actorModelGraph;
    //protected GraphFunction $criticModelGraph;
    //protected ?array $criticTrainableVariables=null;

    public function __construct(
        object $la,
        object $nn,
        array $stateShape,
        int $numActions,
        NDArray $lowerBound,
        NDArray $upperBound,
        ?Policy $policy=null,
        ?int $batchSize=null,
        ?NDArray $mean=null,
        float|NDArray|null $stdDev=null,
        ?NDArray $xInitial=null,
        ?float $gamma=null,
        ?float $theta=null,
        ?float $dt=null,
        ?int $targetUpdatePeriod=null,
        ?float $targetUpdateTau=null,
        ?Optimizer $criticOptimizer=null,
        ?array $criticOptimizerOpts=null,
        ?Optimizer $actorOptimizer=null,
        ?array $actorOptimizerOpts=null,
        //?array $fcLayers=null,
        //?array $staConvLayers=null, ?string $staConvType=null, ?array $staFcLayers=null,
        //?array $actLayers=null,
        //?array $comLayers=null,
        //?float $actorInitMin=null, ?float $actorInitMax=null,
        ?array $actorNetworkOptions=null,
        ?array $criticNetworkOptions=null,
        ?EventManager $eventManager=null,
        ?float $noiseDecay=null,
        float|NDArray|null $minStdDev=null,
        ?bool $episodeAnnealing=null,
        ?object $mo = null,
        )
    {
        $batchSize ??= 32;
        $stdDev ??= 0.2;
        $gamma ??= 0.99;
        $targetUpdatePeriod  ??= 1;
        $targetUpdateTau     ??= 0.005;
        $criticOptimizerOpts ??=  ['lr'=>0.002];
        $actorOptimizerOpts  ??=  ['lr'=>0.001];

        if($lowerBound->shape()!=[$numActions]) {
            throw new InvalidArgumentException(
                "shape of lowerBound must match to the numActions ($numActions): ".
                $la->shapeToString($lowerBound->shape())." givien."
            );
        }
        if($upperBound->shape()!=[$numActions]) {
            throw new InvalidArgumentException(
                "shape of upperBound must match to the numActions ($numActions): ".
                $la->shapeToString($upperBound->shape())." givien."
            );
        }

        $mean ??=  $la->zeros($la->alloc([$numActions]));
        if(is_numeric($stdDev)) {
            $stdDev = $la->fill($stdDev,$la->zeros($la->alloc([$numActions])));
        }
        if(is_numeric($minStdDev)) {
            $minStdDev = $la->fill($minStdDev,$la->zeros($la->alloc([$numActions])));
        }
        $criticOptimizer = $criticOptimizer ?? $nn->optimizers->Adam(...$criticOptimizerOpts);
        $actorOptimizer = $actorOptimizer ?? $nn->optimizers->Adam(...$actorOptimizerOpts);

        $this->mo = $mo;
        $this->actorModel   = $this->buildActorNetwork(
            $nn, $stateShape, $numActions, $actorNetworkOptions,
        );
        $this->criticModel  = $this->buildCriticNetwork(
            $nn, $stateShape, $numActions, $criticNetworkOptions,
        );
        //$this->actorModel->compile(optimizer:$actorOptimizer);
        //$this->criticModel->compile(optimizer:$criticOptimizer);
        $this->targetActor = clone $this->actorModel;
        $this->targetCritic = clone $this->criticModel;

        $policy ??= $this->buildPolicy(
            $la,
            $mean,$stdDev,$lowerBound,$upperBound,
            $theta,$dt,$xInitial,
            $noiseDecay,$minStdDev,$episodeAnnealing,
        );
        parent::__construct($la,$policy,$eventManager);
        $this->nn = $nn;
        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->gamma = $gamma;
        $this->lowerBound = $lowerBound;
        $this->upperBound = $upperBound;
        $this->targetUpdatePeriod = $targetUpdatePeriod;
        $this->targetUpdateTau = $targetUpdateTau;
        $this->criticOptimizer = $criticOptimizer;
        $this->actorOptimizer = $actorOptimizer;
        //$this->backend = $nn->backend();

        //$this->actorTrainableVariables = $this->actorModel->trainableVariables();
        //$this->criticTrainableVariables = $this->criticModel->trainableVariables();

        //$this->actorVariables = $this->actorModel->variables();
        //$this->criticVariables = $this->criticModel->variables();
        //$this->targetActorVariables = $this->targetActor->variables();
        //$this->targetCriticVariables = $this->targetCritic->variables();


        //$this->actorModelGraph = $nn->gradient->function([$this->actorModel,'forward']);
        //$this->criticModelGraph = $nn->gradient->function([$this->criticModel,'forward']);
        //$this->targetActorGraph = $nn->gradient->function([$this->targetActor,'forward']);
        //$this->targetCriticGraph = $nn->gradient->function([$this->targetCritic,'forward']);
        $this->initialize();
        //$this->actorModelGraph = null;
        //$this->criticModelGraph = null;
    }

    protected function buildActorNetwork(
        Builder $nn,
        array $stateShape, int $numActions,
        ?array $actorNetworkOptions=null,
    )
    {
        $actorNetworkOptions ??= [];
        $network = new ActorNetwork($nn,
            $stateShape, $numActions,
            ...$actorNetworkOptions,
        );
        $network->build(array_merge([1],$stateShape));
        return $network;
    }

    protected function buildCriticNetwork(
        Builder $nn,
        array $stateShape, int $numActions,
        ?array $criticNetworkOptions=null,
    )
    {
        $criticNetworkOptions ??= [];
        $network = new CriticNetwork(
            $nn,
            $stateShape, $numActions,
            ...$criticNetworkOptions,
        );
        $network->build(array_merge([1],$stateShape),[1,$numActions]);
        return $network;
    }

    protected function buildPolicy(
        object $la,
        NDArray $mean,
        NDArray $stdDeviation,
        NDArray $lowerBound,
        NDArray $upperBound,
        ?float $theta,
        ?float $dt,
        ?NDArray $xInitial,
        ?float $noiseDecay,
        ?NDArray $minStdDev,
        ?bool $episodeAnnealing,
        )
    {
        $policy = new OUNoise(
            $la,
            $mean,
            $stdDeviation,
            $lowerBound,
            $upperBound,
            theta:$theta,
            dt:$dt,
            x_initial:$xInitial,
            noise_decay:$noiseDecay,
            min_std_dev:$minStdDev,
            episodeAnnealing:$episodeAnnealing,
        );
        return $policy;
    }

    public function actorNetwork()
    {
        return $this->actorModel;
    }

    public function targetActorNetwork()
    {
        return $this->targetActor;
    }

    public function criticNetwork()
    {
        return $this->criticModel;
    }

    public function targetCriticNetwork()
    {
        return $this->targetCritic;
    }

    public function summary()
    {
        echo "***** Actor Network *****\n";
        $this->actorModel->summary();
        echo "\n";
        echo "***** Critic Network *****\n";
        $this->criticModel->summary();
    }

    public function fileExists(string $filename) : bool
    {
        $actormodel = sprintf(self::ACTOR_FILENAME,$filename);
        $criticmodel = sprintf(self::CRITIC_FILENAME,$filename);
        if(file_exists($actormodel) && file_exists($criticmodel)) {
            return true;
        } else {
            return false;
        }
    }

    public function saveWeightsToFile(string $filename) : void
    {
        $actormodel = sprintf(self::ACTOR_FILENAME,$filename);
        $criticmodel = sprintf(self::CRITIC_FILENAME,$filename);
        $this->actorModel->saveWeightsToFile($actormodel);
        $this->criticModel->saveWeightsToFile($criticmodel);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $actormodel = sprintf(self::ACTOR_FILENAME,$filename);
        $criticmodel = sprintf(self::CRITIC_FILENAME,$filename);
        $this->actorModel->loadWeightsFromFile($actormodel);
        $this->criticModel->loadWeightsFromFile($criticmodel);
        //$this->trainModelGraph = $nn->gradient->function([$this->trainModel,'forward']);
        //$this->trainableVariables = $this->trainModel->trainableVariables();
        $this->initialize();
    }

    public function syncWeights($tau=null)
    {
        //$this->copyWeights(
        //    $this->targetActorVariables,
        //    $this->actorVariables,
        //    $tau);
        //$this->copyWeights(
        //    $this->targetCriticVariables,
        //    $this->criticVariables,
        //    $tau);

        //$this->copyWeights(
        //    $this->targetActor->variables(),
        //    $this->actorModel->variables(),
        //    $tau);
        //$this->copyWeights(
        //    $this->targetCritic->variables(),
        //    $this->criticModel->variables(),
        //    $tau);

        $this->targetActor->copyWeights($this->actorModel,$tau);
        $this->targetCritic->copyWeights($this->criticModel,$tau);
    }

    //public function copyWeights($target,$source,float $tau=null) : void
    //{
    //    $K = $this->backend;
    //    if($tau===null) {
    //        $tau = 1.0;
    //    }
    //    $tTau = (1.0-$tau);
    //    foreach (array_map(null,$target,$source) as [$targParam,$srcParam]) {
    //        $K->update_scale($targParam->value(),$tTau);
    //        $K->update_add($targParam->value(),$srcParam->value(),$tau);
    //    }
    //}

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
        return $this->actorModel;
    }

    //public function maxQValue(mixed $states) : float
    //{
    //    $la = $this->la;
    //    $states = $this->atleast2d($states);
    //    $actions = $this->actorModel->predict($states);
    //    $qValues = $this->criticModel->predict([$states, $actions]);
    //    $q = $this->la->max($qValues);
    //    return $q;
    //}

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
        $K = $nn->backend(); 
        $g = $nn->gradient();
        $batchSize = $this->batchSize;
        $stateShape = $this->stateShape;
        $numActions = $this->numActions;
        $gamma = $this->gamma;

        if($experience->size()<$batchSize) {
            return 0.0;
        }
        $transition = $experience->last();
        $endEpisode = $transition[4];  // done

        //$state_batch = $la->zeros($la->alloc(array_merge([$batchSize], $stateShape)));
        //$action_batch = $la->zeros($la->alloc([$batchSize, $numActions]));
        //$next_state_batch = $la->zeros($la->alloc(array_merge([$batchSize], $stateShape)));
        //$reward_batch = $la->zeros($la->alloc([$batchSize,1]));
        ////$discounts = $la->zeros($la->alloc([$batchSize]));
//
        //$batch = $experience->sample($batchSize);
        //$i = 0;
        //foreach($batch as $transition) {
        //    [$state,$action,$next_state,$reward,$done,$truncated,$info] = $transition;
        //    //$state_batch[] = $state;
        //    //$action_batch[] = $action;
        //    //$next_state_batch[] = $next_state;
        //    //$reward_batch[] = $reward;
//
        //    if(is_numeric($state)) {
        //        $state_batch[$i][0] = $state;
        //    } else {
        //        $la->copy($state,$state_batch[$i]);
        //    }
        //    if(is_numeric($action)) {
        //        $action_batch[$i][0] = $action;
        //    } else {
        //        $la->copy($action,$action_batch[$i]);
        //    }
        //    if(is_numeric($next_state)) {
        //        $next_state_batch[$i][0] = $next_state;
        //    } else {
        //        $la->copy($next_state,$next_state_batch[$i]);
        //    }
        //    $reward_batch[$i][0] = $reward;
        //    //$discounts[$i] = $done ? 0.0 : 1.0;
        //    $i++;
        //}

        $batch = $experience->sample($batchSize);
        [$obs,$actions,$nextObs,$rewards,$done,$truncated,$info] = $batch;
        $state_batch = $this->extractStateList($obs);
        $next_state_batch = $this->extractStateList($nextObs);

        $state_batch = $la->stack($state_batch);
        $action_batch = $la->stack($actions);
        $next_state_batch = $la->stack($next_state_batch);
        $reward_batch = $la->expandDims($la->array($rewards),axis:-1);

        $gamma = $g->Variable($gamma);
        $state_batch = $g->Variable($state_batch);
        $action_batch = $g->Variable($action_batch);
        $next_state_batch = $g->Variable($next_state_batch);
        $reward_batch = $g->Variable($reward_batch);
        $training = true;//$g->Variable(true);

        $targetActor = $this->targetActor;
        $targetCritic = $this->targetCritic;
        $criticModel = $this->criticModel;
        $actorModel = $this->actorModel;

        //$targetActor = $this->targetActorGraph;
        //$targetCritic = $this->targetCriticGraph;
        //$criticModel = $this->criticModelGraph;
        //$actorModel = $this->actorModelGraph;


        [$critic_loss,$actionValues] = $nn->with($tape=$g->GradientTape(),function () use (
                $g,$targetActor,$targetCritic,$criticModel,
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,
            ) {
                $target_actions = $targetActor($next_state_batch, $training);
                $target_q_values = $targetCritic($next_state_batch, $target_actions, $training);

                $td_targets = $g->add($reward_batch,$g->mul($gamma,$target_q_values));

                $actionValues = $criticModel($state_batch, $action_batch, $training);

                $critic_loss = $g->reduceMean($g->square($g->sub($td_targets,$actionValues)));
                return [$critic_loss,$actionValues];
            }
        );
        $criticTrainableVariables = $this->criticModel->trainableVariables();
        $critic_grad = $tape->gradient($critic_loss, $criticTrainableVariables);
        $this->criticOptimizer->update($criticTrainableVariables,$critic_grad);
        //echo $K->toString($actionValues,null,true)."\n";



        [$actor_loss,$critic_value] = $nn->with($tape=$g->GradientTape(),function () use (
                $g,$actorModel,$criticModel,
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,
            ) {
                $actions = $actorModel($state_batch, $training);
                $critic_value = $criticModel($state_batch, $actions, $training);
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                $actor_loss = $g->mul($g->Variable(-1),$g->reduceMean($critic_value));
                return [$actor_loss,$critic_value];
            }
        );
        $actorTrainableVariables = $this->actorModel->trainableVariables();
        $actor_grad = $tape->gradient($actor_loss, $actorTrainableVariables);
        $this->actorOptimizer->update($actorTrainableVariables,$actor_grad);
        //echo $K->toString($critic_value,null,true)."\n";


/*
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        if($this->criticModelGraph===null) {
            $this->criticModelGraph = $g->function(function (
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,
            ) use (
                $g,$targetActor,$targetCritic,$criticModel,
            ) {
                $target_actions = $targetActor($next_state_batch, $training);
                $y = $g->add($reward_batch,$g->mul($gamma,
                    $targetCritic($next_state_batch, $target_actions, $training)));
                $critic_value = $criticModel($state_batch, $action_batch, $training);
                $critic_loss = $g->reduceMean($g->square($g->sub($y,$critic_value)));
                return $critic_loss;
            });
        }
        $critic_loss = $nn->with($tape=$g->GradientTape(),
            args:[
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,
            ],
            without_ctx:true,
            func:$this->criticModelGraph,
        );

        $critic_grad = $tape->gradient($critic_loss, $this->criticTrainableVariables);
        $this->criticOptimizer->update($this->criticTrainableVariables,$critic_grad);

        if($this->actorModelGraph===null) {
            $this->actorModelGraph=$g->function(function(
                $state_batch,$training,
            ) use (
                $g,$actorModel,$criticModel,
            ) {
                $actions = $actorModel($state_batch, $training);
                $critic_value = $criticModel($state_batch, $actions, $training);
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                $actor_loss = $g->mul($g->Variable(-1),$g->reduceMean($critic_value));
                return $actor_loss;
            });
        }
        $actor_loss = $nn->with($tape=$g->GradientTape(),
            args:[
                $state_batch,$training,
            ],
            without_ctx:true,
            func:$this->actorModelGraph,
        );
        $actor_grad = $tape->gradient($actor_loss, $this->actorTrainableVariables);
        $this->actorOptimizer->update($this->actorTrainableVariables,$actor_grad);
*/

        if($this->enabledShapeInspection) {
            $this->targetActor->setShapeInspection(false);
            $this->targetCritic->setShapeInspection(false);
            $this->actorModel->setShapeInspection(false);
            $this->criticModel->setShapeInspection(false);
            $this->enabledShapeInspection = false;
        }
        
        $this->updateTarget($endEpisode);

        $loss = $K->scalar($actor_loss->value());
        if($this->metrics->isAttracted('loss')) {
            $this->metrics->update('loss',$loss);
        }
        return $loss;
    }
}
