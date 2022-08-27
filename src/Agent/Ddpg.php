<?php
namespace Rindow\RL\Agents\Agent;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Policy\OUNoise;
use Rindow\RL\Agents\Network\ActorNetwork;
use Rindow\RL\Agents\Network\CriticNetwork;

use InvalidArgumentException;

class Ddpg extends AbstractAgent
{
    const ACTOR_FILENAME = '%s-actor.model';
    const CRITIC_FILENAME = '%s-critic.model';
    protected $la;
    protected $policy;
    protected $gamma;
    protected $obsSize;
    protected $actionSize;
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
    protected $nn;

    public function __construct(
        object $la,
        object $nn,
        array $obsSize,
        array $actionSize,
        NDArray $lower_bound,
        NDArray $upper_bound,
        int $batchSize=null,
        NDArray $mean=null,
        float|NDArray $std_dev=null,
        NDArray $x_initial=null,
        float $gamma=null,
        float $theta=null,
        float $dt=null,
        int $targetUpdatePeriod=null,
        float $targetUpdateTau=null,
        object $criticOptimizer=null,
        array $criticOptimizerOpts=null,
        object $actorOptimizer=null,
        array $actorOptimizerOpts=null,
        array $fcLayers=null,
        array $obsFcLayers=null,
        array $actFcLayers=null,
        array $conFcLayers=null,
        float $actorInitMin=null, float $actorInitMax=null,
        object $mo = null,
        )
    {
        $batchSize = $batchSize ?? 32;
        $std_dev = $std_dev ?? 0.2;
        $gamma = $gamma ?? 0.99;
        $targetUpdatePeriod = $targetUpdatePeriod ?? 1;
        $targetUpdateTau    = $targetUpdateTau ?? 0.005;
        $criticOptimizerOpts = $criticOptimizerOpts ?? ['lr'=>0.002];
        $actorOptimizerOpts  = $actorOptimizerOpts ?? ['lr'=>0.001];

        $mean = $mean ?? $la->zeros($la->alloc($actionSize));
        if(is_numeric($std_dev)) {
            $std_dev = $la->fill($std_dev,$la->zeros($la->alloc($actionSize)));
        }
        $criticOptimizer = $criticOptimizer ?? $nn->optimizers->Adam(...$criticOptimizerOpts);
        $actorOptimizer = $actorOptimizer ?? $nn->optimizers->Adam(...$actorOptimizerOpts);

        $this->la = $la;
        $this->obsSize = $obsSize;
        $this->actionSize = $actionSize;
        $this->batchSize = $batchSize;
        $this->gamma = $gamma;
        $this->lower_bound = $lower_bound;
        $this->upper_bound = $upper_bound;
        $this->targetUpdatePeriod = $targetUpdatePeriod;
        $this->targetUpdateTau = $targetUpdateTau;
        $this->criticOptimizer = $criticOptimizer;
        $this->actorOptimizer = $actorOptimizer;
        $this->mo = $mo;
        $this->nn = $nn;
        $this->backend = $nn->backend();
        $this->actor_model   = $this->buildActorNetwork($obsSize, $actionSize,
             fcLayers:$fcLayers, minval:$actorInitMin, maxval:$actorInitMax);
        $this->critic_model  = $this->buildCriticNetwork($obsSize, $actionSize,
            obsFcLayers:$obsFcLayers,actFcLayers:$actFcLayers,conFcLayers:$conFcLayers);
        $this->actor_model->compile(optimizer:$actorOptimizer);
        $this->critic_model->compile(optimizer:$criticOptimizer);
        $this->target_actor  = $this->buildActorNetwork($obsSize, $actionSize,
            fcLayers:$fcLayers, minval:$actorInitMin, maxval:$actorInitMax);
        $this->target_critic  = $this->buildCriticNetwork($obsSize, $actionSize,
            obsFcLayers:$obsFcLayers,actFcLayers:$actFcLayers,conFcLayers:$conFcLayers);

        $this->policy = $this->buildPolicy(
            $this->actor_model,$mean,$std_dev,$lower_bound,$upper_bound,
            $theta,$dt,$x_initial);

        //$this->actorTrainableVariables = $this->actor_model->trainableVariables();
        //$this->criticTrainableVariables = $this->critic_model->trainableVariables();

        //$this->actorVariables = $this->actor_model->variables();
        //$this->criticVariables = $this->critic_model->variables();
        //$this->targetActorVariables = $this->target_actor->variables();
        //$this->targetCriticVariables = $this->target_critic->variables();


        //$this->actorModelGraph = $nn->gradient->function([$this->actor_model,'forward']);
        //$this->criticModelGraph = $nn->gradient->function([$this->critic_model,'forward']);
        //$this->targetActorGraph = $nn->gradient->function([$this->target_actor,'forward']);
        //$this->targetCriticGraph = $nn->gradient->function([$this->target_critic,'forward']);
        $this->initialize();
        $this->actor_model_graph = null;
        $this->critic_model_graph = null;
    }

    protected function buildActorNetwork(
        array $obsSize, array $actionSize,
        array $convLayers=null,string $convType=null,array $fcLayers=null,
        $activation=null,$kernelInitializer=null,
        float $minval=null, float $maxval=null
    )
    {
        $network = new ActorNetwork($this->la,$this->nn,
            $obsSize, $actionSize,
            convLayers:$convLayers,convType:$convType,fcLayers:$fcLayers,
            activation:$activation,kernelInitializer:$kernelInitializer,
            minval:$minval, maxval:$maxval,
            );
        $network->build(array_merge([1],$obsSize),true);
        return $network;
    }

    protected function buildCriticNetwork(
        array $obsSize, array $actionSize,
        array $obsConvLayers=null,string $obsConvType=null,array $obsFcLayers=null,
        array $actConvLayers=null,string $actConvType=null,array $actFcLayers=null,
        array $conConvLayers=null,string $conConvType=null,array $conFcLayers=null,
        string $activation=null, string $kernelInitializer=null
    )
    {
        $network = new CriticNetwork($this->la,$this->nn,
            $obsSize, $actionSize,
            obsConvLayers: $obsConvLayers, obsConvType:$obsConvType, obsFcLayers:$obsFcLayers,
            actConvLayers: $actConvLayers, actConvType:$actConvType, actFcLayers:$actFcLayers,
            conConvLayers: $conConvLayers, conConvType:$conConvType, conFcLayers:$conFcLayers,
            activation: $activation,       kernelInitializer:$kernelInitializer
            );
        $network->build(array_merge([1],$obsSize),array_merge([1],$actionSize),true);
        return $network;
    }

    protected function buildPolicy(
        $network,$mean,$std_deviation,$lower_bound,$upper_bound,
        $theta,$dt,$x_initial)
    {
        $policy = new OUNoise(
            $this->la, $network,
            $mean,
            $std_deviation,
            $lower_bound,
            $upper_bound,
            theta:$theta,
            dt:$dt,
            x_initial:$x_initial);
        return $policy;
    }

    public function actorNetwork()
    {
        return $this->actor_model;
    }

    public function targetActorNetwork()
    {
        return $this->target_actor;
    }

    public function criticNetwork()
    {
        return $this->critic_model;
    }

    public function targetCriticNetwork()
    {
        return $this->target_critic;
    }

    public function summary()
    {
        echo "***** Actor Network *****\n";
        $this->actor_model->summary();
        echo "\n";
        echo "***** Critic Network *****\n";
        $this->critic_model->summary();
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
        $this->actor_model->saveWeightsToFile($actormodel);
        $this->critic_model->saveWeightsToFile($criticmodel);
    }

    public function loadWeightsFromFile(string $filename) : void
    {
        $actormodel = sprintf(self::ACTOR_FILENAME,$filename);
        $criticmodel = sprintf(self::CRITIC_FILENAME,$filename);
        $this->actor_model->loadWeightsFromFile($actormodel);
        $this->critic_model->loadWeightsFromFile($criticmodel);
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
        //    $this->target_actor->variables(),
        //    $this->actor_model->variables(),
        //    $tau);
        //$this->copyWeights(
        //    $this->target_critic->variables(),
        //    $this->critic_model->variables(),
        //    $tau);

        $this->target_actor->copyWeights($this->actor_model,$tau);
        $this->target_critic->copyWeights($this->critic_model,$tau);
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
        $actions = $this->policy->action($observation,$training,$this->elapsedTime);
        return $actions;
    }

    public function getQValue($observation) : float
    {
        $la = $this->la;
        if(is_numeric($observation)) {
            $observation = $this->la->array([$observation]);
        } elseif(!($observation instanceof NDArray)) {
            throw new InvalidArgumentException('Observation must be NDArray');
        }
        $observation = $la->expandDims($observation,$axis=0);
        $actions = $this->actor_model->predict($observation);
        $qValues = $this->critic_model->predict([$observation, $actions]);
        $q = $this->la->max($qValues);
        return $q;
    }

    public function update($experience) : float
    {
        $la = $this->la;
        $nn = $this->nn; 
        $K = $this->backend; 
        $g = $nn->gradient();
        $batchSize = $this->batchSize;
        $obsSize = $this->obsSize;
        $actionSize = $this->actionSize;
        $gamma = $this->gamma;

        if($experience->size()<$batchSize) {
            return 0.0;
        }
        $state_batch = $la->zeros($la->alloc(array_merge([$batchSize], $obsSize)));
        $action_batch = $la->zeros($la->alloc(array_merge([$batchSize], $actionSize)));
        $next_state_batch = $la->zeros($la->alloc(array_merge([$batchSize], $obsSize)));
        $reward_batch = $la->zeros($la->alloc([$batchSize,1]));
        //$discounts = $la->zeros($la->alloc([$batchSize]));

        $batch = $experience->sample($batchSize);
        $i = 0;
        foreach($batch as $transition) {
            [$state,$action,$next_state,$reward,$done,$info] = $transition;
            //$state_batch[] = $state;
            //$action_batch[] = $action;
            //$next_state_batch[] = $next_state;
            //$reward_batch[] = $reward;

            if(is_numeric($state)) {
                $state_batch[$i][0] = $state;
            } else {
                $la->copy($state,$state_batch[$i]);
            }
            if(is_numeric($action)) {
                $action_batch[$i][0] = $action;
            } else {
                $la->copy($action,$action_batch[$i]);
            }
            if(is_numeric($next_state)) {
                $next_state_batch[$i][0] = $next_state;
            } else {
                $la->copy($next_state,$next_state_batch[$i]);
            }
            $reward_batch[$i][0] = $reward;
            //$discounts[$i] = $done ? 0.0 : 1.0;
            $i++;
        }

        $gamma = $g->Variable($gamma);
        $state_batch = $g->Variable($state_batch);
        $action_batch = $g->Variable($action_batch);
        $next_state_batch = $g->Variable($next_state_batch);
        $reward_batch = $g->Variable($reward_batch);
        $training = $g->Variable(true);

        $target_actor = $this->target_actor;
        $target_critic = $this->target_critic;
        $critic_model = $this->critic_model;
        $actor_model = $this->actor_model;

        //$target_actor = $this->targetActorGraph;
        //$target_critic = $this->targetCriticGraph;
        //$critic_model = $this->criticModelGraph;
        //$actor_model = $this->actorModelGraph;


        [$critic_loss,$q_values] = $nn->with($tape=$g->GradientTape(),function () use (
                $g,$target_actor,$target_critic,$critic_model,
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,
            ) {
                $target_actions = $target_actor($next_state_batch, $training);
                $target_q_values = $target_critic($next_state_batch, $target_actions, $training);

                $td_targets = $g->add($reward_batch,$g->mul($gamma,$target_q_values));

                $q_values = $critic_model($state_batch, $action_batch, $training);

                $critic_loss = $g->reduceMean($g->square($g->sub($td_targets,$q_values)));
                return [$critic_loss,$q_values];
            }
        );
        $this->criticTrainableVariables = $this->critic_model->trainableVariables();
        $critic_grad = $tape->gradient($critic_loss, $this->criticTrainableVariables);
        $this->criticOptimizer->update($this->criticTrainableVariables,$critic_grad);
        //echo $K->toString($q_values,null,true)."\n";



        [$actor_loss,$critic_value] = $nn->with($tape=$g->GradientTape(),function () use (
                $g,$actor_model,$critic_model,
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,
            ) {
                $actions = $actor_model($state_batch, $training);
                $critic_value = $critic_model($state_batch, $actions, $training);
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                $actor_loss = $g->mul($g->Variable(-1),$g->reduceMean($critic_value));
                return [$actor_loss,$critic_value];
            }
        );
        $this->actorTrainableVariables = $this->actor_model->trainableVariables();
        $actor_grad = $tape->gradient($actor_loss, $this->actorTrainableVariables);
        $this->actorOptimizer->update($this->actorTrainableVariables,$actor_grad);
        //echo $K->toString($critic_value,null,true)."\n";


/*
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        if($this->critic_model_graph===null) {
            $this->critic_model_graph = $g->function(function (
                $next_state_batch,$reward_batch,$gamma,
                $state_batch,$action_batch,$training,
            ) use (
                $g,$target_actor,$target_critic,$critic_model,
            ) {
                $target_actions = $target_actor($next_state_batch, $training);
                $y = $g->add($reward_batch,$g->mul($gamma,
                    $target_critic($next_state_batch, $target_actions, $training)));
                $critic_value = $critic_model($state_batch, $action_batch, $training);
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
            func:$this->critic_model_graph,
        );

        $critic_grad = $tape->gradient($critic_loss, $this->criticTrainableVariables);
        $this->criticOptimizer->update($this->criticTrainableVariables,$critic_grad);

        if($this->actor_model_graph===null) {
            $this->actor_model_graph=$g->function(function(
                $state_batch,$training,
            ) use (
                $g,$actor_model,$critic_model,
            ) {
                $actions = $actor_model($state_batch, $training);
                $critic_value = $critic_model($state_batch, $actions, $training);
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
            func:$this->actor_model_graph,
        );
        $actor_grad = $tape->gradient($actor_loss, $this->actorTrainableVariables);
        $this->actorOptimizer->update($this->actorTrainableVariables,$actor_grad);
*/

        if($this->enabledShapeInspection) {
            $this->target_actor->setShapeInspection(false);
            $this->target_critic->setShapeInspection(false);
            $this->actor_model->setShapeInspection(false);
            $this->critic_model->setShapeInspection(false);
            $this->enabledShapeInspection = false;
        }
        if($this->targetUpdatePeriod > 0) {
            $this->targetUpdateTimer--;
            if($this->targetUpdateTimer <= 0) {
                $this->syncWeights($this->targetUpdateTau);
                $this->targetUpdateTimer = $this->targetUpdatePeriod;
            }
        }
        
        $loss = $K->scalar($actor_loss->value());
        return $loss;
    }
}
