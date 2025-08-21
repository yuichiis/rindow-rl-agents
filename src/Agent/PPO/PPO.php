<?php
namespace Rindow\RL\Agents\Agent\PPO;

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

class PPO extends AbstractAgent
{
    const MODEL_FILENAME = '%s.model';
    protected bool $continuous;
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
    protected ?float $clipnorm;
    protected bool $clipValueLoss;
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
    protected NDArray $actionMin;
    protected NDArray $actionMax;

    public function __construct(
        object $la,
        ?bool $continuous=null,
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
        ?float $clipnorm=null,
        ?bool $clipValueLoss=null,
        ?Loss $lossFunc=null,
        ?array $stateShape=null, ?int $numActions=null,
        ?array $fcLayers=null,
        mixed $actionKernelInitializer=null,
        mixed $criticKernelInitializer=null,
        ?float $policyTau=null,?float $policyMin=null,?float $policyMax=null,
        ?Space $actionSpace=null,
        ?string $stateField=null,
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
            $la,$nn,$continuous,$stateShape,$numActions,
            $fcLayers,
            $actionKernelInitializer,
            $criticKernelInitializer,
            $policyMin,$policyMax,
        );
        if(!($network instanceof Estimator)) {
            echo get_class($network);
            throw new InvalidArgumentException('Network must have Network and Estimator interfaces.');
        }
        $policy ??= $this->buildPolicy($la,$continuous,$policyTau,$policyMin,$policyMax);
        parent::__construct($la,policy:$policy,stateField:$stateField);

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
        $clipValueLoss ??= true;

        if(($rolloutSteps % $batchSize) != 0) {
            throw new InvalidArgumentException("rolloutSteps must be divisible by batchSize.: rolloutSteps=$rolloutSteps, batchSize=$batchSize");
        }
        $nn ??= $network->builder();
        $optimizerOpts ??= [];
        $optimizerOpts['lr'] ??= 3e-4;
        $optimizerOpts['epsilon'] ??= 1e-8;
        $optimizer ??= $nn->optimizers->Adam(...$optimizerOpts);
        //$optimizer ??= $nn->optimizers->RMSprop(...$optimizerOpts);
        //$lossFunc ??= $nn->losses()->Huber();
        $lossFunc ??= $nn->losses()->MeanSquaredError();

        $this->continuous = $continuous;
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
        $this->clipValueLoss = $clipValueLoss;
        $this->normAdv = $normAdv;
        $this->optimizer = $optimizer;
        $this->optimizerOpts = $optimizerOpts;
        $this->clipnorm = $clipnorm;
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
        object $la,
        ?Builder $nn,
        bool $continuous,
        ?array $stateShape,
        ?int $numActions,
        ?array $fcLayers,
        mixed $actionKernelInitializer,
        mixed $criticKernelInitializer,
        float|NDArray|null $min,
        float|NDArray|null $max,
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
            $la,$nn,
            $stateShape, $numActions,
            fcLayers:$fcLayers,
            continuous:$continuous,
            actionMin:$min,actionMax:$max,
            actionKernelInitializer:$actionKernelInitializer,
            criticKernelInitializer:$criticKernelInitializer,
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

    public function estimator() : Estimator
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
    public function compute_advantages_and_returns(
        NDArray $rewards, // (rolloutSteps)
        NDArray $values,  // (rolloutSteps+1,1)
        array $dones,
        array $truncated,
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
            if($dones[$t]||$truncated[$t]) {
                // 終端状態の場合、next_valueは0
                $delta = $la->axpy($values[$t], $la->copy($rewards[$t]), alpha:-1);
                $last_advantage = $delta;
            } else {
                // GAEのデルタ: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
                // Rindow-math-matrixの演算は少し複雑なので、要素ごとの演算で分かりやすく書く
                $next_value = $values[$t+1];
                $current_value = $values[$t];
                $reward_t = $rewards[$t];
                
                // $delta = $reward_t + $gamma * $next_value - $current_value;
                $delta = $la->copy($reward_t);
                $delta = $la->axpy($next_value, $delta, $gamma);
                $delta = $la->axpy($current_value, $delta, -1.0);
            
                // GAE: A_t = delta_t + gamma * lambda * A_{t+1}
                // $last_advantage = $delta + $gamma * $gaeLambda * $last_advantage;
                $last_advantage = $la->axpy($last_advantage, $la->copy($delta), $gamma * $gaeLambda);
            }
            $la->copy($last_advantage, $advantages[$t]);
        }

        $returns = $la->axpy($values[R(0,$rolloutSteps)], $la->copy($advantages));

        $advantages = $la->squeeze($advantages,axis:-1);    // (rolloutSteps)
        $returns = $la->squeeze($returns,axis:-1);          // (rolloutSteps)

        return [$advantages, $returns];
    }

    protected function clip_by_global_norm(array $arrayList, float $clipNorm) : array
    {
        $la = $this->la;

        // global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
        $globalNorm = 0.0;
        foreach($arrayList as $array) {
            $globalNorm += ($la->nrm2($array)**2);
        }
        $globalNorm = sqrt($globalNorm);

        //t_list[i] * clip_norm / max(global_norm, clip_norm)
        $scale = $clipNorm/max($globalNorm,$clipNorm);
        $newList = [];
        foreach($arrayList as $array) {
            $newList[] = $la->scal($scale,$la->copy($array));
        }
        return $newList;
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

    public function update(ReplayBuffer $experience) : float
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
        $numActions = $this->numActions;
        $model = $this->model;

        //$states = $la->alloc(array_merge([$rolloutSteps], $stateShape));
        //$rewards = $la->zeros($la->alloc([$rolloutSteps]));
        //if(!$this->continuous) {
        //    $actions = $la->zeros($la->alloc([$rolloutSteps],NDArray::int32));
        //} else {
        //    $actions = $la->zeros($la->alloc([$rolloutSteps,$numActions]));
        //}
        //$dones = [];

        // Rollout
        //$history = $experience->recently($experience->size());
        //$i = 0;
        //foreach ($history as $transition) {
        //    [$state,$action,$nextState,$reward,$done,$truncated,$info] = $transition;
//
        //    // states
        //    if(!($state instanceof NDArray)) {
        //        throw new LogicException("state must be NDArray.");
        //    }
        //    if($la->isInt($state)) {
        //        $state = $la->astype($state,dtype:NDArray::float32);
        //    }
        //    $la->copy($state,$states[$i]);
        //    if(!($action instanceof NDArray)) {
        //        throw new LogicException("action must be NDArray.");
        //    }
        //    // actions
        //    if(!$this->continuous) {
        //        if($action->ndim()!==0) {
        //            $shapeString = $la->shapeToString($action->shape());
        //            throw new LogicException("shape of action must be scalar ndarray. $shapeString given.");
        //        }
        //        $la->copy($action->reshape([1]),$actions[R($i,$i+1)]);
        //    } else {
        //        if($action->ndim()!==1) {
        //            $shapeString = $la->shapeToString($action->shape());
        //            throw new LogicException("shape of action must be rank 1 ndarray. $shapeString given.");
        //        }
        //        $la->copy($action,$actions[$i]);
        //    }
        //    $dones[$i] = ($done || $truncated);
        //    $rewards[$i] = $reward;
        //    $i++;
        //}

        $history = $experience->recently($experience->size());
        [$obs,$actions,$nextObs,$rewards,$dones,$truncated,$info] = $history;
        $states = $this->extractStateList($obs);
        $masks = $this->extractMaskList($obs);
        $nextStates = $this->extractStateList($nextObs);

        $states = $la->stack($states);
        if($la->isInt($states)) {
            $states = $la->astype($states,dtype:NDArray::float32);
        }
        if($masks!==null) {
            $masks = $la->stack($masks);
        }
        $actions = $la->stack($actions);
        $rewards = $la->array($rewards);

        $lastDone = $dones[array_key_last($dones)];
        $lastTruncated = $truncated[array_key_last($truncated)];
        if($lastDone||$lastTruncated) {
            $nextValue = $la->array([[0]]);
        } else {
            $lastNextObs = $nextObs[array_key_last($nextObs)];
            $nextState = $this->extractState($lastNextObs);
            if($la->isInt($nextState)) {
                $nextState = $la->astype($nextState,dtype:NDArray::float32);
            }
            if($nextState->ndim()===0) {
                $nextState = $la->expandDims($nextState,axis:0);
            }
            $nextState = $la->expandDims($nextState,axis:0);
            [$tmp,$nextValue] = $model->forward($nextState,false);
        }

        $experience->clear();

        if(!$this->continuous) {
            [$oldLogits, $oldValues] = $model($states,false);
        } else {
            [$oldMeans, $oldValues, $oldLogStd] = $model($states,false);
            if($this->metrics->isAttracted('actmin')) {
                $actmin = $la->min($oldMeans);
                $this->metrics->update('actmin',$actmin);
            }
            if($this->metrics->isAttracted('actmax')) {
                $actmax = $la->max($oldMeans);
                $this->metrics->update('actmax',$actmax);
            }
        }

        $valuesForGAE = $la->concat([$oldValues,$nextValue],axis:0);

        [$advantages, $returns] = $this->compute_advantages_and_returns(
            $rewards,
            $valuesForGAE,
            $dones,
            $truncated,
            $this->gamma,
            $this->gaeLambda,
        );

        if($this->normAdv) {
            // advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)
            $advantages = $this->standardize($advantages);
        }

        if(!$this->continuous) {
            [$oldLogProbs, $dmy] = $this->log_prob_entropy_categorical($oldLogits,$actions);
        } else {
            [$oldLogProbs, $dmy] = $this->log_prob_entropy_continuous($oldMeans,$oldLogStd,$actions);
        }

        //$valueLossWeight = $g->Variable($this->valueLossWeight);
        //$entropyWeight = $g->Variable($this->entropyWeight);
        $valueLossWeight = $this->valueLossWeight;
        $entropyWeight = $this->entropyWeight;
        $clipEpsilon = $this->clipEpsilon;

        // gradients
        $oldValues = $la->squeeze($oldValues,axis:-1);

        $sourceData = [$states, $actions, $oldLogProbs, $oldValues, $advantages, $returns];
        if($masks!==null) {
            $sourceData[] = $masks;
        }
        $dataset = $nn->data->NDArrayDataset(
            $sourceData,
            batch_size:$batchSize,
            shuffle:true,
        );
        $clipValueLoss = $this->clipValueLoss;
        $model = $this->model;
        $lossFunc = $this->lossFunc;
        $agent = $this;
        $training = $g->Variable(true);
        $loss = 0.0;
        for($epoch=0; $epoch<$this->epochs; $epoch++) {
            foreach($dataset as $batch) {
                if($masks===null) {
                    $masksB = null;
                    [$statesB, $actionsB, $oldLogProbsB, $oldValuesB, $advantagesB, $returnsB] = $batch[0];
                } else {
                    [$statesB, $actionsB, $oldLogProbsB, $oldValuesB, $advantagesB, $returnsB, $masksB] = $batch[0];
                }
                [$totalLoss,$policyLoss,$valueLoss,$entropyLoss] = $nn->with($tape=$g->GradientTape(),function() 
                    use ($la,$agent,$g,$lossFunc,$model,$statesB,$training,$masksB,$actionsB,$oldLogProbsB, 
                        $advantagesB, $returnsB, $valueLossWeight,$entropyWeight,$clipEpsilon,$oldValuesB,
                        $clipValueLoss)
                {
                    if(!$agent->continuous) {
                        [$newLogits, $newValues] = $model($statesB,$training); // (batchSize,numActions),(batchSize,1)
                        $newValues = $g->squeeze($newValues,axis:-1);
                        if($masksB!==null) {
                            $newLogits = $g->masking($masksB,$newLogits,fill:-1e9);
                        }
                        [$newLogProbs, $entropy] = $agent->log_prob_entropy_categorical($newLogits,$actionsB);
                    } else {
                        [$newMeans, $newValues, $newLogStd] = $model($statesB,$training);
                        $newValues = $g->squeeze($newValues,axis:-1);
                        [$newLogProbs, $entropy] = $agent->log_prob_entropy_continuous($newMeans,$newLogStd,$actionsB);
                        // 多次元行動も考慮し、log_probとentropyをスカラーに変換
                        $newLogProbs = $g->reduceSum($newLogProbs, axis:-1);
                        $oldLogProbsB = $g->reduceSum($oldLogProbsB, axis:-1);
                        $entropy = $g->reduceSum($entropy, axis:-1);
                    }
                    // policy loss (Actor Loss)
                    $ratio = $g->exp($g->sub($newLogProbs,$oldLogProbsB));
                    $clippedRatio = $g->clipByValue($ratio, 1-$clipEpsilon, 1+$clipEpsilon);
                    $policyLoss = $g->scale(-1,$g->reduceMean(
                        $g->minimum(
                            $g->mul($ratio,$advantagesB),
                            $g->mul($clippedRatio,$advantagesB),
                        ),
                    ));

                    // Value loss (Critic Loss)
                    if($clipValueLoss) {
                        // Value loss (Clippingを適用したバージョン)
                        // 1. まず、元のValue Lossを計算
                        $valueLossUnclipped = $g->square($g->sub($newValues, $returnsB));
                        // 2. 更新前の価値(oldValues)を基準に、新しい価値(newValues)の変動を制限(clip)する
                        $valuesClipped = $g->add(
                            $oldValuesB,
                            $g->clipByValue(
                                $g->sub($newValues, $oldValuesB),
                                -$clipEpsilon,
                                $clipEpsilon
                            )
                        );
                        // 3. Clipされた価値でのLossを計算
                        $valueLossClipped = $g->square($g->sub($valuesClipped, $returnsB));
                        // 4. 2つのLossのうち、大きい方を選択し、平均を取る
                        $valueLoss = $g->scale(0.5, $g->reduceMean($g->maximum($valueLossUnclipped, $valueLossClipped)));
                    } else {
                        // SB3ではMSE固定
                        //$valueLoss = $lossFunc($returnsB,$newValues);
                        $valueLoss = $g->reduceMean($g->square($g->sub($returnsB,$newValues)));
                    }

                    // policy entropy
                    $entropyLoss = $g->scale(-1,$g->reduceMean($entropy));

                    #echo "actor=".$la->toString($policyLoss,format:'%+3.3f').
                    #    ",critic=".$la->toString($valueLoss,format:'%7.1f').
                    #    ",entropy=".$la->toString($entropyLoss,format:'%+3.3f').
                    #    "\n";
                    // total loss
                    $totalLoss = $g->add(
                        $policyLoss,
                        $g->add(
                            $g->scale($valueLossWeight, $valueLoss),
                            $g->scale($entropyWeight, $entropyLoss)
                        )
                    );
                    #echo $policyLoss->toArray().",".$valueLoss->toArray().",".$entropyLoss->toArray()."\n";
                    return [$totalLoss,$policyLoss,$valueLoss,$entropyLoss];
                });
                $grads = $tape->gradient($totalLoss,$this->trainableVariables);
                if($this->clipnorm!==null) {
                    $grads = $this->clip_by_global_norm($grads,$this->clipnorm);
                }
                $this->optimizer->update($this->trainableVariables,$grads);
                $totalLoss = $K->scalar($totalLoss);
                if($this->metrics->isAttracted('loss')) {
                    $this->metrics->update('loss',$totalLoss);
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
                    $std = $la->reduceMean($la->exp($la->copy($this->model->getLogStd())));
                    $std = $la->scalar($std);
                    $this->metrics->update('std',$std);
                }
                $loss += $totalLoss;
            }
        }

        //echo "loss=".$loss."\n";
        return $loss;
    }

}
