<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Spaces\Box;
use Interop\Polite\AI\RL\Spaces\Space;
use Interop\Polite\AI\RL\Environment as Env;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Gradient\GraphFunction;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Policy\NormalDistribution;
use Rindow\RL\Agents\Agent\AbstractAgent;

use InvalidArgumentException;

class SAC extends AbstractAgent
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
    protected Model $targetCritic;
    protected Variable $logAlpha;
    protected float $targetEntropy;
    protected int $batchSize;
    protected int $startSteps;
    protected NDArray $lowerBound;
    protected NDArray $upperBound;
    protected float $targetUpdateTau;
    protected Optimizer $criticOptimizer;
    protected Optimizer $actorOptimizer;
    protected ?Optimizer $alphaOptimizer;
    //protected GraphFunction $actorModelGraph;
    //protected GraphFunction $criticModelGraph;
    //protected ?array $criticTrainableVariables=null;

    public function __construct(
        object $la,
        object $nn,
        array $stateShape,
        ?int $numActions=null,
        ?NDArray $lowerBound=null,
        ?NDArray $upperBound=null,
        ?Policy $policy=null,
        ?int $batchSize=null,
        ?int $startSteps=null,
        ?float $gamma=null,
        ?float $initailAlpha=null,
        ?bool $autoTuneAlpha=null,
        ?float $targetEntropy=null,
        ?int $targetUpdatePeriod=null,
        ?float $targetUpdateTau=null,
        ?float $learningRate=null,
        ?Optimizer $actorOptimizer=null,
        ?array $actorOptimizerOpts=null,
        ?Optimizer $criticOptimizer=null,
        ?array $criticOptimizerOpts=null,
        ?Optimizer $alphaOptimizer=null,
        ?array $alphaOptimizerOpts=null,
        ?array $fcLayers=null,
        //?array $staConvLayers=null, ?string $staConvType=null, ?array $staFcLayers=null,
        //?array $actLayers=null,
        //?array $comLayers=null,
        //?float $actorInitMin=null, ?float $actorInitMax=null,
        ?array $actorNetworkOptions=null,
        ?array $criticNetworkOptions=null,
        ?EventManager $eventManager=null,
        ?Space $actionSpace=null,
        ?object $mo = null,
        )
    {
        if($actionSpace!==null) {
            if(!($actionSpace instanceof Box)) {
                throw new InvalidArgumentException('type of actionSpace must be Box for continuous actions.');
            }
            $shape = $actionSpace->shape();
            if(count($shape)!==1) {
                throw new InvalidArgumentException('shape of actionSpace must be rank 1.');
            }
            $lowerBound ??= $actionSpace->low();
            $upperBound ??= $actionSpace->high();
            $numActions ??= $shape[0];
        }

        $batchSize ??= 32;
        $startSteps ??= $batchSize;
        $gamma ??= 0.99;
        $initailAlpha ??= 0.2;
        $autoTuneAlpha ??= true;
        $targetUpdatePeriod  ??= 1;
        $targetUpdateTau     ??= 0.005;
        $learningRate ??= 3e-4;
        $criticOptimizerOpts ??=  ['lr'=>$learningRate];
        $actorOptimizerOpts  ??=  ['lr'=>$learningRate];
        $alphaOptimizerOpts  ??=  ['lr'=>$learningRate];

        if($batchSize > $startSteps) {
            $startSteps = $batchSize; 
        }

        if($numActions===null) {
            throw new InvalidArgumentException("Either numActions or actionSpace must be specified.");
        }
        if($lowerBound===null || $lowerBound->shape()!=[$numActions]) {
            throw new InvalidArgumentException(
                "shape of lowerBound must match to the numActions ($numActions): ".
                $la->shapeToString($lowerBound->shape())." givien."
            );
        }
        if($upperBound === null || $upperBound->shape()!=[$numActions]) {
            throw new InvalidArgumentException(
                "shape of upperBound must match to the numActions ($numActions): ".
                $la->shapeToString($upperBound->shape())." givien."
            );
        }

        $criticOptimizer ??= $nn->optimizers->Adam(...$criticOptimizerOpts);
        $actorOptimizer  ??= $nn->optimizers->Adam(...$actorOptimizerOpts);

        $this->mo = $mo;
        $this->actorModel   = $this->buildActorNetwork(
            $nn, $stateShape, $numActions,
            $lowerBound, $upperBound,
            $actorNetworkOptions,
        );
        $this->criticModel  = $this->buildCriticNetwork(
            $nn, $stateShape, $numActions, $criticNetworkOptions,
        );
        //$this->actorModel->compile(optimizer:$actorOptimizer);
        //$this->criticModel->compile(optimizer:$criticOptimizer);
        $this->targetCritic = clone $this->criticModel;

        $g = $nn->gradient();
        if($autoTuneAlpha) {
            $logAlpha = $g->Variable($la->array([log($initailAlpha)], dtype:NDArray::float32), trainable:true);
            $alphaOptimizer  ??= $nn->optimizers->Adam(...$alphaOptimizerOpts);
        } else {
            $logAlpha = $g->Variable($la->array([log($initailAlpha)], dtype:NDArray::float32), trainable:false);
            $alphaOptimizer = null;
        }
        $targetEntropy ??= -$numActions;
        
        parent::__construct($la,$policy);
        $this->nn = $nn;
        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->startSteps = $startSteps;
        $this->gamma = $gamma;
        $this->targetEntropy = $targetEntropy;
        $this->lowerBound = $lowerBound;
        $this->upperBound = $upperBound;
        $this->targetUpdatePeriod = $targetUpdatePeriod;
        $this->targetUpdateTau = $targetUpdateTau;
        $this->criticOptimizer = $criticOptimizer;
        $this->actorOptimizer = $actorOptimizer;
        $this->alphaOptimizer = $alphaOptimizer;

        $this->logAlpha = $logAlpha;
        //$this->backend = $nn->backend();

        //$this->actorTrainableVariables = $this->actorModel->trainableVariables();
        //$this->criticTrainableVariables = $this->criticModel->trainableVariables();

        //$this->actorVariables = $this->actorModel->variables();
        //$this->criticVariables = $this->criticModel->variables();
        //$this->targetCriticVariables = $this->targetCritic->variables();


        //$this->actorModelGraph = $nn->gradient->function([$this->actorModel,'forward']);
        //$this->criticModelGraph = $nn->gradient->function([$this->criticModel,'forward']);
        //$this->targetCriticGraph = $nn->gradient->function([$this->targetCritic,'forward']);
        $this->initialize();
        //$this->actorModelGraph = null;
        //$this->criticModelGraph = null;
    }

    protected function buildActorNetwork(
        Builder $nn,
        array $stateShape, int $numActions,
        NDArray $lowerBound, NDArray $upperBound,
        ?array $actorNetworkOptions=null,
    )
    {
        $actorNetworkOptions ??= [];
        $actorNetworkOptions['lowerBound'] ??= $lowerBound;
        $actorNetworkOptions['upperBound'] ??= $upperBound;
        $network = new ActorNetwork(
            $nn,
            $stateShape,
            $numActions,
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
        NDArray $lowerBound,
        NDArray $upperBound,
        )
    {
        $policy = new NormalDistribution(
            $la,
            min:$lowerBound,
            max:$upperBound,
        );
        return $policy;
    }

    public function actorNetwork()
    {
        return $this->actorModel;
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
        //    $this->targetCriticVariables,
        //    $this->criticVariables,
        //    $tau);

        //$this->copyWeights(
        //    $this->targetCritic->variables(),
        //    $this->criticModel->variables(),
        //    $tau);

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
        return $this->startSteps;
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

    protected function doPolicyActions(NDArray $states,bool $training,?NDArray $masks) : NDArray
    {
        $la = $this->la;
        $actorModel = $this->actorModel;
        if($training) {
            [$actions,$logProb,$logStd] = $actorModel($states,training:false, deterministic:false);
        } else {
            $actions = $actorModel($states,training:false, deterministic:true);
        }
        $actions = $la->copy($actions);
        $actions = $la->maximum($actions,$this->lowerBound);
        $actions = $la->minimum($actions,$this->upperBound);
        return $actions;
    }

    public function collect(
        Env $env,
        ReplayBuffer $experience,
        int $step,
        int $episodeSteps,
        NDArray|array $states,
        ?array $info,
        ) : array
    {
        $la = $this->la;
        if($step < $this->startSteps) {
            $actions = $la->multiply(
                $this->upperBound,
                $la->randomUniform([$this->numActions],-1.0,1.0),
            );
        } else {
            $actions = $this->action($states,training:true,info:$info);
        }
        [$nextState,$reward,$done,$truncated,$info] = $env->step($actions);
        $nextState = $this->customState($env,$nextState,$done,$truncated,$info);
        $reward = $this->customReward($env,$episodeSteps,$states,$actions,$nextState,$reward,$done,$truncated,$info);
        $experience->add([$states,$actions,$nextState,$reward,$done,$truncated,$info]);
        return [$nextState,$reward,$done,$truncated,$info];
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

        $batch = $experience->sample($batchSize);
        [$obs,$actions,$nextObs,$rewards,$done,$truncated,$info] = $batch;
        $state_batch = $this->extractStateList($obs);
        $next_state_batch = $this->extractStateList($nextObs);

        $state_batch = $la->stack($state_batch);
        $action_batch = $la->stack($actions);
        $next_state_batch = $la->stack($next_state_batch);
        $reward_batch = $la->expandDims($la->array($rewards),axis:-1);
        $done_batch = $la->array($done);

        $gamma = $g->Variable($gamma);
        $state_batch = $g->Variable($state_batch);
        $action_batch = $g->Variable($action_batch);
        $next_state_batch = $g->Variable($next_state_batch);
        $reward_batch = $g->Variable($reward_batch);
        //$done_batch = $g->Variable($done_batch);
        $discount_batch = $la->expandDims($la->increment($la->copy($done_batch),beta:1,alpha:-1),axis:-1);
        $training = true;//$g->Variable(true);
        $targetEntropy = $g->Variable($this->targetEntropy);

        $targetCritic = $this->targetCritic;
        $criticModel = $this->criticModel;
        $actorModel = $this->actorModel;
        $logAlpha = $this->logAlpha;

        $alpha = $la->exp($la->copy($this->logAlpha));
        $critic_loss = $nn->with($tape=$g->GradientTape(),function () use (
                $g,$actorModel,$targetCritic,$criticModel,
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,$alpha,$discount_batch,
            ) {
                [$next_actions, $next_log_prob, $logStd] = $actorModel($next_state_batch, $training);
                [$target_q1_value_next,$target_q2_value_next] = $targetCritic($next_state_batch, $next_actions, $training);
                $target_q_next = $g->minimum($target_q1_value_next, $target_q2_value_next);

                $soft_target = $g->sub($target_q_next, $g->mul($alpha,$next_log_prob));
                $y = $g->add($reward_batch, $g->scale($gamma, $g->mul($discount_batch, $soft_target)));
                [$q1_current, $q2_current] = $criticModel($state_batch, $action_batch);

                $critic_loss = $g->add(
                    $g->reduceMean($g->square($g->sub($y, $q1_current))),
                    $g->reduceMean($g->square($g->sub($y, $q2_current))),
                );
                return $critic_loss;
            }
        );
        $criticTrainableVariables = $this->criticModel->trainableVariables();
        $critic_grad = $tape->gradient($critic_loss, $criticTrainableVariables);
        $this->criticOptimizer->update($criticTrainableVariables,$critic_grad);
        //echo $K->toString($actionValues,null,true)."\n";

        [$actor_loss,$pi_log_prob,$logStd] = $nn->with($tape=$g->GradientTape(),function () use (
                $g,$actorModel,$criticModel,
                $gamma,
                $state_batch,$training,$alpha,
            ) {
                [$pi_action, $pi_log_prob, $logStd] = $actorModel($state_batch);
                [$q1_pi, $q2_pi] = $criticModel($state_batch, $pi_action);
                $min_q_pi = $g->minimum($q1_pi, $q2_pi);
                $actor_loss = $g->reduceMean($g->sub($g->mul($alpha, $pi_log_prob), $min_q_pi));

                return [$actor_loss,$pi_log_prob,$logStd];
            }
        );
        $actorTrainableVariables = $this->actorModel->trainableVariables();
        $actor_grad = $tape->gradient($actor_loss, $actorTrainableVariables);
        $this->actorOptimizer->update($actorTrainableVariables,$actor_grad);


        $alpha_loss = $nn->with($tape=$g->GradientTape(),function () use (
                $g,$logAlpha,$pi_log_prob,$targetEntropy,
            ) {
                $alpha_loss = $g->scale(-1,$g->reduceMean($g->mul($logAlpha, $g->add($g->stopGradient($pi_log_prob), $targetEntropy))));
                return $alpha_loss;
            }
        );
        if($this->alphaOptimizer) {
            $alpha_grad = $tape->gradient($alpha_loss, [$logAlpha]);
            $this->alphaOptimizer->update([$logAlpha],$alpha_grad);
        }

        if($this->enabledShapeInspection) {
            $this->targetCritic->setShapeInspection(false);
            $this->actorModel->setShapeInspection(false);
            $this->criticModel->setShapeInspection(false);
            $this->enabledShapeInspection = false;
        }
        
        $this->updateTarget($endEpisode);

        $actor_loss = $K->scalar($actor_loss->value());
        if($this->metrics->isAttracted('Ploss')) {
            $this->metrics->update('Ploss',$actor_loss);
        }
        if($this->metrics->isAttracted('Vloss')) {
            $critic_loss = $K->scalar($critic_loss->value());
            $this->metrics->update('Vloss',$critic_loss);
        }
        if($this->metrics->isAttracted('Aloss')) {
            $alpha_loss = $K->scalar($alpha_loss->value());
            $this->metrics->update('Aloss',$alpha_loss);
        }
        if($this->metrics->isAttracted('alpha')) {
            $alpha = $K->scalar($la->reduceMean($alpha));
            $this->metrics->update('alpha',$alpha);
        }
        if($this->metrics->isAttracted('std')) {
            $std = $la->sum($la->exp($la->copy($logStd)))/$logStd->size();
            $std = $la->scalar($std);
            $this->metrics->update('std',$std);
        }
        return $actor_loss;
    }
}
