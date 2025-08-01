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
        $optimizerOpts ??= ['lr'=>7e-4];
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
     * GAEとリターンを計算する (変更なし)
     */
    protected function compute_advantages_and_returns(
        NDArray $rewards,
        NDArray $values,
        array $dones,
        float $gamma,
        float $gaeLambda,
        ) : array
    {
        $la = $this->la;

        $advantages = $la->zerosLike($rewards);
        $last_advantage = 0.0;

        $batchSize = count($rewards);
        for($t=$batchSize-1;$t>=0;$t--) {
            if($dones[$t]) {
                $delta = $rewards[$t] - $values[$t][0];
                //$delta = $la->axpy($values[$t],$la->copy($rewards[$t]),alpha:-1);
                $last_advantage = $delta;
            } else {
                $delta = $rewards[$t] + $gamma * $values[$t+1][0] - $values[$t][0];
                //$delta =  $la->axpy(
                //    $la->axpy($values[$t+1],$la->copy($values[$t]),alpha:-$gamma),
                //    $la->copy($rewards[$t]),
                //    alpha:-1
                //);
                $last_advantage = $delta + $gamma * $gaeLambda * $last_advantage;
                //$last_advantage = $la->axpy(
                //    $last_advantage,
                //    $delta,
                //    alpha:($gamma * $gaeLambda),
                //);
            }
            $advantages[$t] = $last_advantage;
        }
        // broadcast add
        $returns = $la->add($advantages, $la->squeeze($values[[0,$batchSize]],axis:-1));
        return [$advantages, $returns];
    }

    protected function computeGAE(
        NDArray $rewards, 
        NDArray $values,
        NDArray $next_values,
        NDArray $dones,
        float $gamma,
        float $lambda_gae,
        ) : NDArray
    {
        $advantages = $g->zerosLike($rewards);
        $lastAdvantage = 0.0;
        $batchSize = count($rewards);
        for($t=$batchSize-1;$t>=0;$t--) {
            $mask = 1.0 - $dones[$t];
            $delta = $rewards[$t] + $gamma * $next_values[$t] * $mask - $values[$t];
            $advantages[$t] = $delta + $gamma * $lambda_gae * $last_advantage * $mask;
            $last_advantage = $advantages[$t];
        }
        return $advantages;
    }

    # === 正規分布関数 ===
    /**
     * 正規分布からサンプリング
     */
    protected function sample(
        NDArray $logits,
        ) : NDArray
    {
        $samples = $this->categorical($logits, num_samples:1);
        return tf.squeeze($samples, axis:-1);
    }

    /**
     *  正規分布の統計量を計算する。
     *  Args:
     *      mu (tf.Tensor): 平均
     *      log_std (tf.Tensor): Logされた標準偏差
     *      actions (tf.Tensor): 確率を計算したいアクション
     *  Returns:
     *      tuple[tf.Tensor, tf.Tensor]: (log_prob, entropy)
     */
    protected function calculate_normal_dist_stats(
        NDArray $mu,
        NDArray $log_std,
        NDArray $actions,
        ) : NDArray
    {
        $log_prob = (
            -$log_std
            -0.5 * $g->log(2.0 * $g->pi())
            -0.5 * $g->square(($actions - $mu) / $g->exp($log_std))
        );
        $log_prob = $g->reduce_sum($log_prob, axis:1, keepdims:true);
        $entropy = 0.5 + 0.5 * $g->log(2 * $g->pi()) + $log_std;
        #entropy = $g->reduce_sum($entropy, axis:1, keepdims:true);

        return [$log_prob, $entropy];
    }

    # --- 1. 公式の sparse_softmax_cross_entropy_with_logits を使用 ---
    protected function sparse_softmax_cross_entropy_with_logits(
        NDArray $labels,
        NDArray $logits,
    ) : NDArray
    {
        # --- 2. 基本的な関数で手動で再現 ---
        
        # Step 1: Log-Sum-Expの計算 (数値的安定性のための重要なステップ)
        # tf.reduce_logsumexpは log(sum(exp(logits))) を安定して計算する関数
        #$log_sum_exp = $g->reduce_logsumexp($logits, axis:-1);
        
        # Step 2: 正解ラベルをワンホットベクトルに変換
        # labels: [0, 2, 3] -> one_hot_labels: [[1,0,0,0], [0,0,1,0], [0,0,0,1]]
        $one_hot_labels = $g->oneHot($labels, numClass:4);
        
        # Step 3: 正解クラスのlogit値を取得
        # l_i,y_i に相当する部分を計算
        # one_hot_labelsとlogitsの要素ごとの積を取り、和を計算することで実現
        $correct_class_logits = $g->reduceSum($g->mul($one_hot_labels,$logits), axis:-1);
        return $correct_class_logits;

        ## Step 4: 最終的な損失を計算
        ## L_i = log_sum_exp - l_i,y_i
        #$loss = $log_sum_exp - $correct_class_logits;
        #return $loss;
    }

    /**
     *   対数確率(log_prob)とエントロピーを計算します。
     *   tfp.distributions.Categorical の log_prob() と entropy() の代替です。
     * 
     *   Args:
     *     logits: 形状が [batch_size, num_actions] のテンソル。
     *             分布の正規化されていない対数確率。
     *     actions: 形状が [batch_size] のテンソル。
     *              対数確率を計算したいアクション（カテゴリのインデックス）。
     * 
     *   Returns:
     *     log_prob: 形状が [batch_size] のテンソル。各アクションの対数確率。
     *     entropy: 形状が [batch_size] のテンソル。各分布のエントロピー。
     */
    protected function log_prob_entropy_analog(
        NDArray $logits,
        NDArray $actions,
        ) : array
    {
        # --- 1. 対数確率 (Log Probability) の計算 ---
        # tf.nn.sparse_softmax_cross_entropy_with_logits は、指定されたaction (labels)
        # の「負の」対数確率を効率的かつ数値的に安定して計算します。
        # log_prob = -cross_entropy
        $negative_log_prob = $this->sparse_softmax_cross_entropy_with_logits(
            labels:actions, logits:logits);
        $log_prob = $g->scale(-1,$negative_log_prob);
      
        # --- 2. エントロピー (Entropy) の計算 ---
        # エントロピーの定義: H(p) = - Σ_i p_i * log(p_i)
        # 数値的安定性のため、softmaxとlog_softmaxをそれぞれ使用します。
        $probs = $g->softmax($logits, axis:-1);
        $log_probs = $g->log($g->softmax($logits, axis:-1));
        
        # ゼロ確率の項 (p_i=0) は log(p_i) が-infになりますが、
        # p_i * log(p_i) は 0 * -inf = NaN となります。
        # しかし、softmaxの結果が厳密に0になることはまれで、
        # TFの計算では適切に処理されるため、通常は問題ありません。
        $entropy = $g->scale(-1,$g->reduceSum($g->mul($probs,$log_probs), axis:-1));
      
        return [$log_prob, $entropy];
    }

    protected function standardize(NDArray $x, ?bool $ddof=null) : NDArray
    {
        $ddof ??= false;

        $la = $this->la;

        // baseline
        $mean = $la->reduceMean($x,axis:0);
        $baseX = $la->add($mean,$la->copy($x),alpha:-1.0,trans:true);

        // std
        if($ddof) {
            $n = $x->size()-1;
        } else {
            $n = $x->size();
        }
        $variance = $la->scal(1/$n, $la->reduceSum($la->square($baseX),axis:0));
        $stdDev = $la->sqrt($variance);

        // standardize
        $result = $la->multiply($la->reciprocal($stdDev,beta:1e-8),$baseX);

        return $result;
    }

    protected function log_prob_entropy(
        NDArray $logits,
        NDArray $actions,
    ) : array
    {
        $g = $this->g;
        $log_probs_all = $g->log($g->softmax($logits));
        $selected_log_probs = $g->gather($log_probs_all, $actions, batchDims:1);

        $probs = $g->softmax($logits);
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
            //$advantages = $la->multiply($la->add($la->reduceMean($advantages),$la->copy($advantages), alpha:-1), $la->reciprocal($la->std($advantages),1e-8));
            $advantages = $this->standardize($advantages);
        }

        #onehot_actions = tf.one_hot(actions, model.output[0].shape[1])

        [$oldLogProbs, $dmy] = $this->log_prob_entropy($oldLogits,$actions);

        $valueLossWeight = $g->Variable($this->valueLossWeight);
        $entropyWeight = $g->Variable($this->entropyWeight);
        $clipEpsilon = $this->clipEpsilon;

        // gradients
        $dataset = $nn->data->NDArrayDataset(
            [$states, $actions, $oldLogProbs, $advantages, $returns],
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
                [$statesB, $actionsB, $oldLogProbsB, $advantagesB, $returnsB] = $batch[0];
                [$totalLoss,$entropyLoss] = $nn->with($tape=$g->GradientTape(),function() 
                    use ($ppo,$g,$lossFunc,$model,$statesB,$training,$actionsB,$oldLogProbsB,
                        $advantagesB, $returnsB, $valueLossWeight,$entropyWeight,$clipEpsilon)
                {
                    [$newLogits, $newValues] = $model($statesB,$training);
                    $newValues = $g->squeeze($newValues,axis:-1);

                    // policy loss
                    [$newLogProbs, $entropy] = $ppo->log_prob_entropy($newLogits,$actionsB);
                    $ratio = $g->exp($g->sub($newLogProbs,$oldLogProbsB));
                    $clippedRatio = $g->clipByValue($ratio, 1-$clipEpsilon, 1+$clipEpsilon);
                    $policyLoss = $g->scale(-1,
                        $g->minimum(
                            $g->mul($ratio,$advantagesB),
                            $g->mul($clippedRatio,$advantagesB),
                        ),
                    );

                    // Value loss
                    // SB3ではMSE固定
                    $valueLoss = $lossFunc($returnsB,$newValues);

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

                    return [$totalLoss,$entropyLoss];
                });
                $grads = $tape->gradient($totalLoss,$this->trainableVariables);
                $this->optimizer->update($this->trainableVariables,$grads);
                $totalLoss = $K->scalar($la->reduceSum($totalLoss));
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
