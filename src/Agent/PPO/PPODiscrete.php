<?php
namespace Rindow\RL\Agents\Agent\PPO;

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

class PPO extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';
    protected float $gamma;
    protected float $gaeLambda;
    protected float $valueLossWeight;
    protected float $entropyWeight;
    protected float $clipEpsilon;
    protected bool $normAdv;
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
    protected int $epochs;
    protected int $rolloutSteps;

    public function __construct(
        object $la,
        ?Network $network=null,
        ?Policy $policy=null,
        ?int $batchSize=null,
        ?int $epochs=null,
        ?int $rolloutSteps=null,
        ?float $gamma=null,
        ?float $gaeLambda=null,
        ?float $valueLossWeight=null,
        ?float $entropyWeight=null,
        ?float $clipEpsilon=null,
        ?bool $normAdv=null,
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
        $batchSize ??= 64;
        $epochs ??= 10;
        $rolloutSteps ??= 2048;
        $gamma ??= 0.99;
        $gaeLambda ??= 0.95;
        $valueLossWeight ??= 0.5;
        $entropyWeight ??= 0.0;
        $clipEpsilon ??= 0.2;
        $normAdv ??= true;

        if(($rolloutSteps % $batchSize) != 0) {
            throw new InvalidArgumentException('rolloutSteps must be divisible by batchSize.');
        }
        $nn ??= $network->builder();
        $optimizerOpts ??= ['lr'=>3e-4];
        $optimizer ??= $nn->optimizers->Adam(...$optimizerOpts);
        //$optimizer ??= $nn->optimizers->RMSprop(...$optimizerOpts);
        //$lossFunc ??= $nn->losses()->Huber();
        $lossFunc ??= $nn->losses()->MeanSquaredError();

        $this->stateShape = $stateShape;
        $this->numActions = $numActions;
        $this->batchSize = $batchSize;
        $this->epochs = $epochs;
        $this->rolloutSteps = $rolloutSteps;
        $this->gamma = $gamma;
        $this->gaeLambda = $gaeLambda;
        $this->valueLossWeight = $valueLossWeight;
        $this->entropyWeight = $entropyWeight;
        $this->clipEpsilon = $clipEpsilon;
        $this->normAdv = $normAdv;
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
        return false;
    }

    public function subStepLength() : int
    {
        return $this->rolloutSteps;
    }

    public function numRolloutSteps() : int
    {
        return $this->rolloutSteps;
    }

    protected function estimator() : Estimator
    {
        return $this->model;
    }

    protected function updateTarget($endEpisode)
    {
    }

    /**
     * Computes Generalized Advantage Estimation (GAE).
     * GAEとリターンを計算する
     */
    protected function compute_advantages_and_returns(
        NDArray $rewards, // (rolloutSteps)
        NDArray $values,  // (rolloutSteps+1,1)
        array $dones,
        float $gamma,
        float $gaeLambda,
        ) : array
    {
        $la = $this->la;
        $rewards = $la->expandDims($rewards,axis:-1);   // (rolloutSteps,1)

        $advantages = $la->zerosLike($rewards);         // (rolloutSteps,1)
        $last_advantage = $la->zeros($la->alloc([1]));  // (1)

        $rolloutSteps = count($rewards);
        for($t=$rolloutSteps-1;$t>=0;$t--) {
            if($dones[$t]) {
                //$delta = $rewards[$t] - $values[$t];
                $delta = $la->axpy($values[$t],$la->copy($rewards[$t]), alpha:-1);
                $last_advantage = $delta;
            } else {
                //$delta = $rewards[$t] + $gamma * $values[$t+1] - $values[$t];
                //$delta = $rewards[$t] - (-$gamma * $values[$t+1] + $values[$t]);
                //$delta = -(-$gamma * $values[$t+1] + $values[$t]) + $rewards[$t];
                $delta = $la->axpy(
                    $la->axpy(
                        $values[$t+1],
                        $la->copy($values[$t]),
                        alpha:-$gamma
                    ),
                    $la->copy($rewards[$t]),
                    alpha:-1
                );
                //$last_advantage = $delta + $gamma * $gaeLambda * $last_advantage;
                //$last_advantage = $gamma * $gaeLambda * $last_advantage + $delta;
                $last_advantage = $la->axpy($last_advantage, $delta,alpha:$gamma*$gaeLambda);
            }
            $la->copy($last_advantage,$advantages[$t]);
        }
        $returns = $la->axpy($values[R(0,$rolloutSteps)], $advantages);

        $advantages = $la->squeeze($advantages,axis:-1);    // (rolloutSteps)
        $returns = $la->squeeze($returns,axis:-1);          // (rolloutSteps)

        return [$advantages, $returns];
    }

    protected function standardize(
        NDArray $x,         // (rolloutSteps)
        ?bool $ddof=null
        ) : NDArray
    {
        $ddof ??= false;

        $la = $this->la;

        // baseline
        $mean = $la->reduceMean($x,axis:0);     // ()
        $baseX = $la->add($mean,$la->copy($x),alpha:-1.0);  // (rolloutSteps)

        // std
        if($ddof) {
            $n = $x->size()-1;
        } else {
            $n = $x->size();
        }
        $variance = $la->scal(1/$n, $la->reduceSum($la->square($la->copy($baseX)),axis:0)); // ()
        $stdDev = $la->sqrt($variance); // ()

        // standardize
        $result = $la->multiply($la->reciprocal($stdDev,beta:1e-8),$baseX); // (rolloutSteps)
        return $result; // (rolloutSteps)
    }

    protected function log_prob_entropy(
        NDArray $logits,    // (batchsize,numActions) : float32
        NDArray $actions,   // (numActions) : int32
    ) : array
    {
        $g = $this->g;
        $log_probs_all = $g->log($g->softmax($logits)); // (batchsize,numActions) : float32
        $selected_log_probs = $g->gather($log_probs_all, $actions, batchDims:1); // (batchsize) : float32

        $probs = $g->softmax($logits);  // (batchsize,numActions)
        $entropy = $g->scale(-1,$g->reduceSum($g->mul($probs, $log_probs_all), axis:1));

        return [$selected_log_probs, $entropy];
    }

    public function update($experience) : float
    {
        if($experience->size()<$this->rolloutSteps) {
            return 0.0;
        }

        $la = $this->la;
        $nn = $this->nn;
        $g  = $this->g;
        $K  = $nn->backend();
        $stateShape = $this->stateShape;
        $batchSize = $this->batchSize;
        $rolloutSteps = $this->rolloutSteps;

        $states = $la->alloc(array_merge([$rolloutSteps], $stateShape));
        $rewards = $la->zeros($la->alloc([$rolloutSteps]));
        $actions = $la->zeros($la->alloc([$rolloutSteps],NDArray::int32));
        $dones = [];

        // Rollout
        $history = $experience->recently($experience->size());
        $totalReward = 0;
        $last_advantage = 0.0;
        $i = 0;
        foreach ($history as $transition) {
            [$state,$action,$nextState,$reward,$done,$truncated,$info] = $transition;

            // states
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
            // actions
            if($action->ndim()!==0) {
                throw new LogicException("shape of action must be scalar ndarray.");
            }
            $la->copy($action->reshape([1]),$actions[R($i,$i+1)]);
            $dones[$i] = ($done || $truncated);
            $rewards[$i] = $reward;
            $i++;
        }
        $experience->clear();

        $model = $this->model;
        [$oldLogits, $oldValues] = $model($states,false);
        [$dmy, $nextValue] = $model($la->expandDims($nextState,0),false);

        $valuesForGAE = $la->concat([$oldValues,$nextValue],axis:0);

        [$advantages, $returns] = $this->compute_advantages_and_returns(
            $rewards,
            $valuesForGAE,
            $dones,
            $this->gamma,
            $this->gaeLambda,
        );

        if($this->normAdv) {
            // advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)
            $advantages = $this->standardize($advantages);
        }

        [$oldLogProbs, $dmy] = $this->log_prob_entropy($oldLogits,$actions);

        $valueLossWeight = $g->Variable($this->valueLossWeight);
        $entropyWeight = $g->Variable($this->entropyWeight);
        $clipEpsilon = $this->clipEpsilon;

        // gradients
        $oldValues = $la->squeeze($oldValues,axis:-1);

        $dataset = $nn->data->NDArrayDataset(
            [$states, $actions, $oldLogProbs, $oldValues, $advantages, $returns],
            batch_size:$batchSize,
            shuffle:true,
        );
        $model = $this->model;
        $lossFunc = $this->lossFunc;
        $ppo = $this;
        $training = $g->Variable(true);
        $loss = 0.0;
        for($epoch=0; $epoch<$this->epochs; $epoch++) {
            foreach($dataset as $batch) {
                [$statesB, $actionsB, $oldLogProbsB, $oldValuesB, $advantagesB, $returnsB] = $batch[0];
                [$totalLoss,$entropyLoss] = $nn->with($tape=$g->GradientTape(),function() 
                    use ($la,$ppo,$g,$lossFunc,$model,$statesB,$training,$actionsB,$oldLogProbsB,$oldValuesB,
                        $advantagesB, $returnsB, $valueLossWeight,$entropyWeight,$clipEpsilon)
                {
                    [$newLogits, $newValues] = $model($statesB,$training);
                    $newValues = $g->squeeze($newValues,axis:-1);

                    // policy loss
                    [$newLogProbs, $entropy] = $ppo->log_prob_entropy($newLogits,$actionsB);
                    $ratio = $g->exp($g->sub($newLogProbs,$oldLogProbsB));
                    $clippedRatio = $g->clipByValue($ratio, 1-$clipEpsilon, 1+$clipEpsilon);
                    $policyLoss = $g->scale(-1,$g->reduceMean(
                        $g->minimum(
                            $g->mul($ratio,$advantagesB),
                            $g->mul($clippedRatio,$advantagesB),
                        ),
                    ));

                    // Value loss
                    // SB3ではMSE固定
                    $valueLoss = $lossFunc($returnsB,$newValues);
                    //$valueLoss = $g->reduceMean($g->square($g->sub($returnsB,$newValues)));

                    //// Value loss (Clippingを適用したバージョン)
                    //// 1. まず、元のValue Lossを計算
                    //$valueLossUnclipped = $g->square($g->sub($newValues, $returnsB));
                    //// 2. 更新前の価値(oldValues)を基準に、新しい価値(newValues)の変動を制限(clip)する
                    //$valuesClipped = $g->add(
                    //    $oldValuesB,
                    //    $g->clipByValue(
                    //        $g->sub($newValues, $oldValuesB),
                    //        -$clipEpsilon,
                    //        $clipEpsilon
                    //    )
                    //);
                    //// 3. Clipされた価値でのLossを計算
                    //$valueLossClipped = $g->square($g->sub($valuesClipped, $returnsB));
                    //// 4. 2つのLossのうち、大きい方を選択し、平均を取る
                    //$valueLoss = $g->scale(0.5, $g->reduceMean($g->maximum($valueLossUnclipped, $valueLossClipped)));                    

                    // policy entropy
                    $entropyLoss = $g->scale(-1,$g->reduceMean($entropy));

                    // total loss
                    $totalLoss = $g->add(
                        $policyLoss,
                        $g->add(
                            $g->scale($valueLossWeight, $valueLoss),
                            $g->scale($entropyWeight, $entropyLoss)
                        )
                    );
                    #echo $policyLoss->toArray().",".$valueLoss->toArray().",".$entropyLoss->toArray()."\n";

                    return [$totalLoss,$entropyLoss];
                });
                $grads = $tape->gradient($totalLoss,$this->trainableVariables);
                $this->optimizer->update($this->trainableVariables,$grads);
                $totalLoss = $K->scalar($totalLoss);
                if($this->metrics->isAttracted('loss')) {
                    $this->metrics->update('loss',$totalLoss);
                }
                if($this->metrics->isAttracted('entropy')) {
                    $entropyLoss = $K->scalar($entropyLoss);
                    $this->metrics->update('entropy',$entropyLoss);
                }
                $loss += $totalLoss;
            }
        }

        //echo "loss=".$loss."\n";
        return $loss;
    }

}
