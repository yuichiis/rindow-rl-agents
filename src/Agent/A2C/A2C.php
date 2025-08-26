<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Spaces\Box;
use Interop\Polite\AI\RL\Spaces\Space;
use Interop\Polite\AI\RL\Environment as Env;
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
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Agent\AbstractAgent;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\RL\Agents\Policy\NormalDistribution;
use function Rindow\Math\Matrix\R;

class A2C extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';
    protected bool $continuous;
    protected float $gamma;
    protected float $valueLossWeight;
    protected float $entropyWeight;
    protected bool $useBaseline;
    protected bool $useNormalize;
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
    protected NDArray $actionMin;
    protected NDArray $actionMax;

    public function __construct(
        object $la,
        ?bool $continuous=null,
        ?Network $network=null,
        ?Policy $policy=null,
        ?int $batchSize=null,
        ?float $gamma=null,
        ?float $valueLossWeight=null,
        ?float $entropyWeight=null,
        ?bool $useBaseline=null,
        ?bool $useNormalize=null,
        ?object $nn=null,
        ?object $optimizer=null,
        ?array $optimizerOpts=null,
        ?Loss $lossFunc=null,
        ?array $stateShape=null, ?int $numActions=null,
        ?array $fcLayers=null,
        mixed $actionKernelInitializer=null,
        mixed $criticKernelInitializer=null,
        ?float $policyTau=null,?float $policyMin=null,?float $policyMax=null,
        ?Space $actionSpace=null,
        ?string $stateField=null,
        ?float $initialStd=null,
        ?object $mo = null
        )
    {
        $continuous ??= false;
        if($continuous) {
            if($actionSpace===null) {
                throw new InvalidArgumentException('actionSpace must be specified for continuous actions.');
            }
            if(!($actionSpace instanceof Box)) {
                throw new InvalidArgumentException('type of actionSpace must be Box for continuous actions.');
            }
            $shape = $actionSpace->shape();
            if(count($shape)!==1) {
                throw new InvalidArgumentException('shape of actionSpace must be rank 1.');
            }
            $policyMin ??= $actionSpace->low();
            $policyMax ??= $actionSpace->high();
            $numActions ??= $shape[0];
        }

        $network ??= $this->buildNetwork(
            $nn,$continuous,$stateShape,$numActions,
            $fcLayers,
            $actionKernelInitializer,
            $criticKernelInitializer,
            $policyMin,$policyMax,
            $initialStd,
        );
        if(!($network instanceof Estimator)) {
            echo get_class($network);
            throw new InvalidArgumentException('Network must have Network and Estimator interfaces.');
        }
        $policy ??= $this->buildPolicy($la,$continuous,$policyTau,$policyMin,$policyMax);
        parent::__construct($la,policy:$policy,stateField:$stateField);

        $stateShape ??= $network->stateShape();
        $numActions ??= $network->numActions();
        $batchSize ??= 32;
        $gamma ??= 0.99;
        $valueLossWeight ??= 0.5;
        $entropyWeight ??= 0.0;
        $useBaseline ??= false;
        $useNormalize ??= true;
        $nn ??= $network->builder();
        $optimizerOpts ??= ['lr'=>7e-4];
        $optimizer ??= $nn->optimizers->Adam(...$optimizerOpts);
        //$optimizer ??= $nn->optimizers->RMSprop(...$optimizerOpts);
        //$lossFunc ??= $nn->losses()->Huber();
        $lossFunc ??= $nn->losses()->MeanSquaredError();

        $this->continuous = $continuous;
        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->gamma = $gamma;
        $this->valueLossWeight = $valueLossWeight;
        $this->entropyWeight = $entropyWeight;
        $this->useBaseline = $useBaseline;
        $this->useNormalize = $useNormalize;
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

    protected function buildNetwork(
        Builder $nn,
        bool $continuous,
        ?array $stateShape,
        ?int $numActions,
        ?array $fcLayers,
        mixed $actionKernelInitializer,
        mixed $criticKernelInitializer,
        float|NDArray|null $min,
        float|NDArray|null $max,
        ?float $initialStd,
        )
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
        $network = new ActorCriticNetwork(
            $nn,
            $stateShape, $numActions,
            fcLayers:$fcLayers,
            continuous:$continuous,
            actionMin:$min,actionMax:$max,
            actionKernelInitializer:$actionKernelInitializer,
            criticKernelInitializer:$criticKernelInitializer,
            initialStd:$initialStd,
        );
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
        bool $continuous,
        ?float $tau=null,
        float|NDArray|null $min=null,
        float|NDArray|null $max=null,
        )
    {
        if(!$continuous) {
            // Discrete Actions
            $policy = new Boltzmann(
                $la,
                $tau,
                $min,
                $max,
                fromLogits:true,
            );
        } else {
            // Continuous Actions
            if(!($min instanceof NDArray)) {
                throw new InvalidArgumentException("policyMin must be NDArray for continuous actions.");
            }
            if(!($max instanceof NDArray)) {
                throw new InvalidArgumentException("policyMax must be NDArray for continuous actions.");
            }
            $policy = new NormalDistribution(
                $la,
            );
            $this->actionMin = $min;
            $this->actionMax = $max;
        }
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

    public function numRolloutSteps() : int
    {
        return $this->batchSize;
    }

    protected function estimator() : Estimator
    {
        return $this->model;
    }

    protected function updateTarget($endEpisode)
    {
    }

    protected function compute_discounted_rewards(
        array $rewards,
        float $nextValue,
        array $dones,
        array $truncated,
        float $gamma,
        ) : NDArray
    {
        $discountedRewards = [];
        $discounted = $nextValue;
        for($i=count($rewards)-1; $i>=0; $i--) {
            if($dones[$i]||$truncated[$i]) {
                $discounted = 0;
            }
            $discounted = $rewards[$i] + $discounted*$gamma;
            $discountedRewards[] = $discounted;
        }
        $discountedRewards = array_reverse($discountedRewards);
        $discountedRewards = $this->la->array($discountedRewards);
        return $discountedRewards;
    }

    public function action(array|NDArray $obs, ?bool $training=null, ?array $info=null, ?bool $parallel = null) : NDArray
    {
        $la = $this->la;
        $action = parent::action($obs,training:$training,info:$info,parallel:$parallel);
        if($this->continuous) {
            if($this->actionMin!==null) {
                $action = $la->maximum($la->copy($action),$this->actionMin);
            }
            if($this->actionMax!==null) {
                $action = $la->minimum($la->copy($action),$this->actionMax);
            }
        }
        return $action;
    }

    public function collect(
        Env $env,
        ReplayBuffer $experience,
        int $episodeSteps,
        array|NDArray $obs,
        ?array $info,
        ) : array
    {
        $la = $this->la;
        $actions = parent::action($obs,training:true,info:$info);
        $orignalActions = $actions;
        if($this->continuous) {
            if($this->actionMin!==null) {
                $actions = $la->maximum($la->copy($actions),$this->actionMin);
            }
            if($this->actionMax!==null) {
                $actions = $la->minimum($la->copy($actions),$this->actionMax);
            }
        }
        [$nextObs,$reward,$done,$truncated,$info] = $env->step($actions);
        $nextObs = $this->customState($env,$nextObs,$done,$truncated,$info);
        $reward = $this->customReward($env,$episodeSteps,$obs,$actions,$nextObs,$reward,$done,$truncated,$info);
        $experience->add([$obs,$orignalActions,$nextObs,$reward,$done,$truncated,$info]);
        return [$nextObs,$reward,$done,$truncated,$info];
    }

    public function update(ReplayBuffer  $experience) : float
    {
        $la = $this->la;
        $nn = $this->nn;
        $g  = $this->g;
        $K  = $nn->backend();
        $batchSize = $this->batchSize;
        $stateShape = $this->stateShape;
        $numActions = $this->numActions;
        $model = $this->model;

        if($experience->size()<$batchSize) {
            return 0.0;
        }

        [$obs,$action,$nextObs,$reward,$done,$truncated,$info] = $experience->last();  // done

        if($done||$truncated) {
            $nextValue = 0;
        } else {
            $nextState = $this->extractState($nextObs);
            if($la->isInt($nextState)) {
                $nextState = $la->astype($nextState,dtype:NDArray::float32);
            }
            if($nextState->ndim()) {
                $nextState = $la->expandDims($nextState,axis:0);
            }
            if(!$this->continuous) {
                [$dmy,$nextValue] = $model($nextState,false);
            } else {
                [$dmy,$nextValue,$dmy2] = $model($nextState,false);
            }
            $nextValue = $la->scalar(($la->squeeze($nextValue)));
        }

        $batchSize = $experience->size();
        $history = $experience->recently($batchSize);
        [$obs,$actions,$nextState,$rewards,$dones,$truncated,$info] = $history;

        $states = $this->extractStateList($obs);
        $masks = $this->extractMaskList($obs);
       
        $states = $la->stack($states);
        if($la->isInt($states)) {
            $states = $la->astype($states,dtype:NDArray::float32);
        }
        if($masks!==null) {
            $masks = $la->stack($masks);
        }
        $actions = $la->stack($actions);

        $discountedRewards = $this->compute_discounted_rewards($rewards,$nextValue,$dones,$truncated,$this->gamma);

        $experience->clear();

        // baseline
        if($this->useBaseline) {
            $baseline = $la->reduceMean($discountedRewards);
            $la->add($baseline, $discountedRewards, alpha:-1);
        }

        if(!$this->continuous) {
            [$dmy, $values] = $model($states,false);
        } else {
            [$dmy, $values, $dmy2] = $model($states,false);
        }
        $values = $la->squeeze($values,axis:-1);
        // advantage
        $advantage = $g->sub($discountedRewards,$la->copy($values));
        if($this->useNormalize) {
            // advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)
            $advantage = $this->standardize($advantage);
        }

        $discountedRewards = $g->Variable($discountedRewards);
        $advantage = $g->Variable($advantage);

        $valueLossWeight = $this->valueLossWeight;
        $entropyWeight = $this->entropyWeight;
        // gradients
        $agent = $this;
        $lossFunc = $this->lossFunc;
        $training = $g->Variable(true);
        [$loss, $policyLoss, $valueLoss, $entropyLoss, $logStd] = $nn->with($tape=$g->GradientTape(),function() 
            use ($la,$agent,$g,$lossFunc,$model,$states,$training,$masks,$actions,$discountedRewards,$advantage,
                $valueLossWeight,$entropyWeight)
        {
            // action log_probs
            if(!$agent->continuous) {
                [$logits, $values] = $model($states,$training);
                $values = $g->squeeze($values,axis:-1);
                if($masks!==null) {
                    $logits = $g->masking($masks,$logits,fill:-1e9);
                }
                [$log_probs, $entropy] = $agent->log_prob_entropy_categorical($logits,$actions);
                $logStd = null;
            } else {
                [$means, $values, $logStd] = $model($states,$training);
                $values = $g->squeeze($values,axis:-1);
                [$log_probs, $entropy] = $agent->log_prob_entropy_continuous($means,$logStd,$actions);
                // 多次元行動も考慮し、log_probとentropyをスカラーに変換
                $log_probs = $g->reduceSum($log_probs, axis:-1);
                $entropy = $g->reduceSum($entropy, axis:-1);
            }

            // policy loss
            $policyLoss = $g->scale(-1,$g->reduceMean($g->mul($log_probs,$advantage)));
            
            // Value loss
            // SB3ではMSE固定
            $valueLoss = $lossFunc($discountedRewards,$values);

            // entropy loss
            $entropyLoss = $g->reduceMean($entropy);

            // total loss
            $loss = $g->add(
                $policyLoss,
                $g->sub(
                    $g->scale($valueLossWeight, $valueLoss),
                    $g->scale($entropyWeight, $entropyLoss)
                ),
            );

            return [$loss, $policyLoss, $valueLoss, $entropyLoss, $logStd];
        });
        $grads = $tape->gradient($loss,$this->trainableVariables);
        $this->optimizer->update($this->trainableVariables,$grads);

        $loss = $K->scalar($loss);
        if($this->metrics->isAttracted('loss')) {
            $this->metrics->update('loss',$loss);
        }
        if($this->metrics->isAttracted('Ploss')) {
            $policyLoss = $K->scalar($policyLoss);
            $this->metrics->update('Ploss',$policyLoss);
        }
        if($this->metrics->isAttracted('Vloss')) {
            $valueLoss = $K->scalar($valueLoss);
            $this->metrics->update('Vloss',$valueLoss);
        }
        if($this->metrics->isAttracted('entropy')) {
            $entropyLoss = $K->scalar($entropyLoss);
            $this->metrics->update('entropy',$entropyLoss);
        }
        if($this->metrics->isAttracted('std')) {
            $std = $la->exp($la->reduceMean($logStd));
            $std = $la->scalar($std);
            $this->metrics->update('std',$std);
        }
        return $loss;
    }

}
