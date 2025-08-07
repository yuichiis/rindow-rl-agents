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
use Rindow\RL\Agents\EventManager;
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
        ?Loss $lossFunc=null,
        ?array $stateShape=null, ?int $numActions=null,
        ?array $fcLayers=null,
        ?float $policyTau=null,?float $policyMin=null,?float $policyMax=null,
        ?Space $actionSpace=null,
        ?int $experienceSize=null,
        ?EventManager $eventManager=null,
        ?ReplayBuffer $replayBuffer=null,
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

        $network ??= $this->buildNetwork($la,$nn,$continuous,$stateShape,$numActions,$fcLayers);
        if(!($network instanceof Estimator)) {
            echo get_class($network);
            throw new InvalidArgumentException('Network must have Network and Estimator interfaces.');
        }
        $policy ??= $this->buildPolicy($la,$continuous,$policyTau,$policyMin,$policyMax);
        parent::__construct($la,policy:$policy,experienceSize:$experienceSize,replayBuffer:$replayBuffer);

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
        $network = new ActorCriticNetwork($la,$nn,
            $stateShape, $numActions,fcLayers:$fcLayers,continuous:$continuous,
            actionKernelInitializer:'he_normal',
            criticKernelInitializer:'he_normal',
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
    public function compute_advantages_and_returns(
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
        $returns = $la->axpy($values[R(0,$rolloutSteps)], $la->copy($advantages));

        $advantages = $la->squeeze($advantages,axis:-1);    // (rolloutSteps)
        $returns = $la->squeeze($returns,axis:-1);          // (rolloutSteps)

        return [$advantages, $returns];
    }

    public function standardize(
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

    /**
     *  Args:
     *      mean (tf.Tensor):    平均
     *      logStd (tf.Tensor):  Logされた標準偏差
     *      value (tf.Tensor):   確率を計算したい値
     *  Returns:
     *      tuple[tf.Tensor, tf.Tensor]: (log_prob, entropy)
     */
    protected function log_prob_entropy_continuous(
        NDArray $mean,      // (batchSize,numActions)
        NDArray $logStd,    // (numActions)
        NDArray $value,     // (batchSize,numActions)
    ) : array
    {
        $g = $this->g;
        // log_prob =
        //      -0.5 * tf.square((actions - mu) / tf.math.exp(log_std))
        //      -log_std
        //      -0.5 * np.log(2.0 * np.pi))
        $stableStd = $g->add($g->exp($logStd),$g->constant(1e-8));
        $logProb = $g->sub(
            $g->sub(
                $g->scale(-0.5,$g->square($g->div($g->sub($value,$mean),$stableStd))),
                $g->log($stableStd)
            ),
            $g->constant(0.5 * log(2.0 * pi()))
        );
        $logProb = $g->reduceSum($logProb, axis:1, keepdims:true);
        $entropy = $g->add($g->constant(0.5 + 0.5*log(2*pi())), $g->log($stableStd));
        $entropy = $g->add($g->zerosLike($mean),$entropy); // 他のテンソルとの互換性のため

        return [$logProb, $entropy]; // logProb=(batchsize,numActions), entropy=(1,numActions)
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

    public function reset(Env $env) : array
    {
        [$states,$info] = $env->reset();
        $states = $this->customState($env,$states,false,false,$info);
        return [$states,$info];
    }

    public function step(
        Env $env,
        int $episodeSteps,
        NDArray $states,
        ?bool $training=null,
        ?array $info=null,
        ) : array
    {
        $la = $this->la;
        $training ??= false;
        $actions = $this->action($states,training:$training,info:$info);
        $orignalActions = $actions;
        if($this->continuous) {
            if($this->actionMin!==null) {
                $actions = $la->maximum($la->copy($actions),$this->actionMin);
            }
            if($this->actionMax!==null) {
                $actions = $la->minimum($la->copy($actions),$this->actionMax);
            }
        }
        [$nextState,$reward,$done,$truncated,$info] = $env->step($actions);
        $nextState = $this->customState($env,$nextState,$done,$truncated,$info);
        $reward = $this->customReward($env,$episodeSteps,$states,$actions,$nextState,$reward,$done,$truncated,$info);
        if($training) {
            $this->experience->add([$states,$orignalActions,$nextState,$reward,$done,$truncated,$info]);
        }
        return [$nextState,$reward,$done,$truncated,$info];
    }

    public function update(mixed $experience=null) : float
    {
        $experience ??= $this->experience;
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

        $states = $la->alloc(array_merge([$rolloutSteps], $stateShape));
        $rewards = $la->zeros($la->alloc([$rolloutSteps]));
        if(!$this->continuous) {
            $actions = $la->zeros($la->alloc([$rolloutSteps],NDArray::int32));
        } else {
            $actions = $la->zeros($la->alloc([$rolloutSteps,$numActions]));
        }
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
            if(!$this->continuous) {
                if($action->ndim()!==0) {
                    throw new LogicException("shape of action must be scalar ndarray.");
                }
                $la->copy($action->reshape([1]),$actions[R($i,$i+1)]);
            } else {
                if($action->ndim()!==1) {
                    throw new LogicException("shape of action must be rank 1 ndarray.");
                }
                $la->copy($action,$actions[$i]);
            }
            $dones[$i] = ($done || $truncated);
            $rewards[$i] = $reward;
            $i++;
        }
        $experience->clear();

        $model = $this->model;
        if(!$this->continuous) {
            [$oldLogits, $oldValues] = $model($states,false);
        } else {
            [$oldMeans, $oldValues, $oldLogStd] = $model($states,false);
        }

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

        if(!$this->continuous) {
            [$oldLogProbs, $dmy] = $this->log_prob_entropy($oldLogits,$actions);
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
                [$totalLoss,$policyLoss,$valueLoss,$entropyLoss] = $nn->with($tape=$g->GradientTape(),function() 
                    use ($la,$ppo,$g,$lossFunc,$model,$statesB,$training,$actionsB,$oldLogProbsB, 
                        $advantagesB, $returnsB, $valueLossWeight,$entropyWeight,$clipEpsilon,$oldValuesB) 
                {
                    if(!$ppo->continuous) {
                        [$newLogits, $newValues] = $model($statesB,$training);
                        $newValues = $g->squeeze($newValues,axis:-1);
                        [$newLogProbs, $entropy] = $ppo->log_prob_entropy($newLogits,$actionsB);
                    } else {
                        [$newMeans, $newValues, $newLogStd] = $model($statesB,$training);
                        $newValues = $g->squeeze($newValues,axis:-1);
                        [$newLogProbs, $entropy] = $ppo->log_prob_entropy_continuous($newMeans,$newLogStd,$actionsB);
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
                    // SB3ではMSE固定
                    //$valueLoss = $lossFunc($returnsB,$newValues);
                    $valueLoss = $g->reduceMean($g->square($g->sub($returnsB,$newValues)));

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
                if($this->metrics->isAttracted('entropy')) {
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
