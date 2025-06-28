<?php
namespace Rindow\RL\Agents\Agent\Reinforce;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
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
    protected array $stateShape;
    protected int $numActions;
    protected Loss $lossFn;
    protected Optimizer $optimizer;
    protected $mo;
    protected Builder $nn;
    protected object $g;
    protected Layer $gather;
    protected PolicyNetwork $model;
    protected GraphFunction $trainModelGraph;
    protected array $trainableVariables;


    public function __construct(
        object $la,
        ?Network $network=null,
        ?Policy $policy=null,
        ?float $gamma=null,
        ?bool $useBaseline=null,
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
        ?EventManager $eventManager=null,
        ?object $mo = null
        )
    {
        $network ??= $this->buildNetwork(
            $la,$nn,
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
        parent::__construct($la,$policy,$eventManager);

        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->gamma = $gamma;
        $this->useBaseline = $useBaseline;
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
        object $la,
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
            $la, $nn,
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
        $this->gather = $nn->layers->Gather(axis:-1);
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

    protected function estimator() : Estimator
    {
        return $this->model;
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
        $stateShape = $this->stateShape;
        $numActions = $this->numActions;

        $steps = $experience->size();
        $rewards = $la->alloc([$steps]);
        $states = $la->alloc(array_merge([$steps], $stateShape));
        $discountedRewards = $la->alloc([$steps]);
        $actions = $la->alloc([$steps],NDArray::int32);
        //$discounts = $la->zeros($la->alloc([$steps]));
        $masks = $la->ones($la->alloc([$steps, $numActions],NDArray::bool));

        $history = $experience->recently($steps);

        $totalReward = 0;
        $i = $steps-1;
        $discounted = 0;
        $history = array_reverse($history);
        foreach ($history as $transition) {
            [$state,$action,$nextState,$reward,$done,$truncated,$info] = $transition;
            $discounted = $reward + $discounted*$this->gamma;
            $discountedRewards[$i] = $discounted;
            $rewards[$i] = $reward;
            if(!($state instanceof NDArray)) {
                throw new LogicException("state must be NDArray.");
            }
            if($la->isInt($state)) {
                $state = $la->astype($state,dtype:NDArray::float32);
            }
            $la->copy($state,$states[$i]);
            if($action->ndim()!==0) {
                throw new LogicException("shape of action must be scalar ndarray.");
            }
            $la->copy($action->reshape([1]),$actions[R($i,$i+1)]);
            if($info!=null) {
                if(isset($info['validActions'])) {
                    $la->copy($info['validActions'],$masks[$i]);
                }
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
                use ($g,$gather,$trainModel,$states,$training,$masks,$actions,$discountedRewards) {
            $policyLogits = $trainModel($states,$training);
            $policyLogits = $g->masking($masks,$policyLogits,-1e9);
            $policyProbs = $g->softmax($policyLogits);
            $policyProbs = $gather->forward([$policyProbs,$actions],$training);
            $policyProbs = $g->clipByValue($policyProbs, 1e-10, 1.0);
            $logProbs = $g->log($policyProbs);
            $lossPolicy = $g->mul($logProbs,$discountedRewards);
            $loss = $g->mul($g->Variable(-1),$g->reduceSum($lossPolicy));
            return $loss;
        });
        $grads = $tape->gradient($loss,$this->trainableVariables);
        $this->optimizer->update($this->trainableVariables,$grads);

        $loss = $K->scalar($loss);
        //echo "loss=".$loss."\n";
        return $loss;
    }
}
