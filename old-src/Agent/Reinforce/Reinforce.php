<?php
namespace Rindow\RL\Agents\Agent\Reinforce;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment as Env;
use InvalidArgumentException;
use LogicException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\RL\Agents\Agent\AbstractAgent;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Gradient\GraphFunction;
use function Rindow\Math\Matrix\R;

class Reinforce extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';

    protected float $gamma;
    protected bool $useBaseline;
    protected bool $useNormalize;
    protected array $stateShape;
    protected int $numActions;
    protected Loss $lossFn;
    protected Optimizer $optimizer;
    protected $mo;
    protected Builder $nn;
    protected object $g;
    protected PolicyNetwork $model;
    protected GraphFunction $trainModelGraph;
    protected array $trainableVariables;


    public function __construct(
        object $la,
        ?Network $network=null,
        ?Policy $policy=null,
        ?float $gamma=null,
        ?bool $useBaseline=null,
        ?bool $useNormalize=null,
        ?object $nn=null,
        ?object $lossFn=null,
        ?array $lossOpts=null,
        ?object $optimizer=null,
        ?array $optimizerOpts=null,
        ?array $stateShape=null,
        ?int $numActions=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ?float $boltzTau=null,
        ?string $stateField=null,
        ?object $mo = null
        )
    {
        $network ??= $this->buildNetwork(
            $nn,
            $stateShape,$numActions,$fcLayers,$activation,$kernelInitializer,
        );
        $nn ??= $network->builder();
        if(!($network instanceof Estimator)) {
            throw new InvalidArgumentException('Network must have Network and Estimator interfaces.');
        }
        $policy ??= $this->buildPolicy($la,$boltzTau);
        $stateShape ??= $network->stateShape();
        $numActions ??= $network->numActions();
        $gamma ??= 0.99;
        $useBaseline ??= false;
        $useNormalize ??= false;

        parent::__construct($la,policy:$policy,stateField:$stateField);

        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->gamma = $gamma;
        $this->useBaseline = $useBaseline;
        $this->useNormalize = $useNormalize;
        $this->mo = $mo;

        $this->nn = $nn;
        $this->g = $nn->gradient();
        $this->model = $this->buildTrainingModel(
            $la,$nn,$network,$lossFn,$lossOpts,$optimizer,$optimizerOpts,
        );
        $this->trainModelGraph = $nn->gradient->function([$this->model,'forward']);
        $this->trainableVariables = $this->model->trainableVariables();
        $this->initialize();
    }

    protected function buildNetwork(
        Builder $nn,
        array $stateShape,
        int $numActions,
        ?array $fcLayers,
        ?string $activation=null,
        ?string $kernelInitializer=null,
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
        $network = new PolicyNetwork(
            $nn,
            $stateShape, $numActions,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
        );
        return $network;
    }

    protected function buildPolicy($la,$tau)
    {
        $policy = new Boltzmann(
            $la,
            tau:$tau,
            fromLogits:true,
        );
        return $policy;
    }

    protected function buildTrainingModel(
        object $la,
        Builder $nn,
        Estimator $network,
        ?Loss $lossFn,
        ?array $lossOpts,
        ?Optimizer $optimizer,
        ?array $optimizerOpts,
        )
    {
        $lossOpts ??= [];
        $optimizerOpts ??= [];
        $lossFn ??= $nn->losses->Huber(...$lossOpts);
        $optimizer ??=$nn->optimizers->Adam(...$optimizerOpts);
        $this->lossFn = $lossFn;
        $this->optimizer = $optimizer;
        $network->compile(loss:$this->lossFn,optimizer:$this->optimizer);
        $network->build(array_merge([1],$this->stateShape));

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

    public function initialize() : void // : Operation
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

    public function numRolloutSteps() : int
    {
        return 1;
    }

    protected function estimator() : Estimator
    {
        return $this->model;
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

    /**
    * @param Any $params
    */
    public function update(ReplayBuffer $experience) : float
    {
        $la = $this->la;
        $nn = $this->nn;
        $K = $nn->backend();
        $g  = $this->g;
        $stateShape = $this->stateShape;
        $numActions = $this->numActions;

        //$steps = $experience->size();
        //$rewards = $la->alloc([$steps]);
        //$states = $la->alloc(array_merge([$steps], $stateShape));
        //$discountedRewards = $la->alloc([$steps]);
        //$actions = $la->alloc([$steps],NDArray::int32);
        //$masks = $la->ones($la->alloc([$steps, $numActions],NDArray::bool));
//
        //$history = $experience->recently($steps);
//
        //$totalReward = 0;
        //$i = $steps-1;
        //$discounted = 0;
        //$history = array_reverse($history);
        //foreach ($history as $transition) {
        //    [$obs,$action,$nextState,$reward,$done,$truncated,$info] = $transition;
        //    $state = $this->extractState($obs);
        //    $discounted = $reward + $discounted*$this->gamma;
        //    $discountedRewards[$i] = $discounted;
        //    $rewards[$i] = $reward;
        //    if(!($state instanceof NDArray)) {
        //        throw new LogicException("state must be NDArray.");
        //    }
        //    if($la->isInt($state)) {
        //        $state = $la->astype($state,dtype:NDArray::float32);
        //    }
        //    $la->copy($state,$states[$i]);
        //    if($action->ndim()!==0) {
        //        throw new LogicException("shape of action must be scalar ndarray.");
        //    }
        //    $la->copy($action->reshape([1]),$actions[R($i,$i+1)]);
        //    $mask = $this->extractMask($obs);
        //    if($mask!=null) {
        //        $la->copy($mask,$masks[$i]);
        //    }
        //    $i--;
        //}

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
        } else {
            $masks = $la->ones($la->alloc([$batchSize, $numActions],NDArray::bool));
        }
        $actions = $la->stack($actions);

        $discountedRewards = $this->compute_discounted_rewards($rewards,0,$dones,$truncated,$this->gamma);

        $experience->clear();

        if($this->useBaseline) {
            $baseline = $K->mean($discountedRewards);
            $la->increment($discountedRewards, -1.0*$baseline);
        }
        if($this->useNormalize) {
            $discountedRewards = $this->standardize($discountedRewards);
        }

        $discountedRewards = $g->Variable($discountedRewards);

        $trainModel = $this->model;
        $agent = $this;
        $training = $g->Variable(true);
        $loss = $nn->with($tape=$g->GradientTape(),function() 
                use ($la,$agent,$g,$trainModel,$states,$training,$masks,$actions,$discountedRewards) {
            $policyLogits = $trainModel($states,$training);
            $policyLogits = $g->masking($masks,$policyLogits,fill:-INF);

            $logProbs = $agent->log_prob_categorical($policyLogits,$actions);
            //echo $la->toString($logProbs,format:'%8.6f',indent:true)."\n";
            
            $lossPolicy = $g->mul($logProbs,$discountedRewards);
            $loss = $g->mul($g->Variable(-1),$g->reduceSum($lossPolicy));
            return $loss;
        });
        $grads = $tape->gradient($loss,$this->trainableVariables);
        $this->optimizer->update($this->trainableVariables,$grads);

        $loss = $K->scalar($loss);
        if($this->metrics->isAttracted('loss')) {
            $this->metrics->update('loss',$loss);
        }
        //echo "loss=".$loss."\n";
        return $loss;
    }
}
