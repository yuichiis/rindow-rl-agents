<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;


/** 
 *
 * SAC + gSDE エージェント
 *
 */
class SACGSDEAgent
{
    private const CHECKPOINT_VERSION = 1;
    private Builder $nn;
    private object $la;
    private object $g;
    private int $actDim;
    private int $obsDim;
    private float $actLimit;
    public GSDEActor $actor;
    public Critic $critic;
    public Critic $criticTarget;
    private float $targetEntropy;
    private Variable $logAlpha;
    private object $actorOpt;
    private object $criticOpt;
    private object $alphaOpt;
    private array $lastActorGrads = [];
    private array $lastCriticGrads = [];
    private ?Variable $lastLogPi = null;
    private ?Variable $lastQData = null;
    private ?Variable $lastQPi = null;
    private ?Variable $lastTargetQ = null;
    private float $gamma;
    private float $tau;
    private int $batchSize;

    public function __construct(
        Builder $nn,
        int $obsDim,
        int $actDim,
        float $actLimit,
        int $gsdeLatentDim,
        int $hiddenDim,
        float $lrActor,
        float $lrCritic,
        float $lrAlpha,
        float $alphaInit,
        float $gamma,
        float $tau,
        int $batchSize,
    )
    {
        $this->nn = $nn;
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        $this->actDim   = $actDim;
        $this->obsDim   = $obsDim;
        $this->actLimit = $actLimit;
        $this->gamma = $gamma;
        $this->tau = $tau;
        $this->batchSize = $batchSize;
        $la = $this->la; 

        $this->actor         = new GSDEActor($nn, $obsDim, $actDim, $gsdeLatentDim, $hiddenDim);
        $this->critic        = new Critic($nn, $obsDim, $actDim, $hiddenDim);
        $this->criticTarget = new Critic($nn, $obsDim, $actDim, $hiddenDim);

        // ダミー入力で build してから weights をコピー
        $dummyObs = $this->g->Variable($la->zeros($la->alloc([1, $obsDim])));
        $dummyAct = $this->g->Variable($la->zeros($la->alloc([1, $actDim])));
        
        $this->actor->forwardTrain($dummyObs);
        $this->critic->forward($dummyObs, $dummyAct);
        $this->criticTarget->forward($dummyObs, $dummyAct);
        
        $criticVars = $this->critic->trainableVariables();
        $criticVars = $this->critic->variables();


        $this->softUpdate($this->g, $this->critic, $this->criticTarget, 1.0);  // 完全コピー

        $this->actorOpt  = $nn->optimizers->Adam(lr: $lrActor);
        $this->criticOpt = $nn->optimizers->Adam(lr: $lrCritic);
        $this->alphaOpt  = $nn->optimizers->Adam(lr: $lrAlpha);

        // 自動エントロピー調整
        // PyTorch: torch.tensor(log(ALPHA_INIT), requires_grad=True)
        // TF:      tf.Variable(..., trainable=True)
        $this->targetEntropy = -(float)$actDim;
        $this->logAlpha = $this->g->Variable(
            $this->la->array([log($alphaInit)]),
            trainable:true, name:"log_alpha"
        );
    }

    /**
     * ソフトアップデートユーティリティ
     * 
     * PyTorch:
     *     for p, p_tgt in zip(src.parameters(), tgt.parameters()):
     *         p_tgt.data.copy_(tau * p + (1-tau) * p_tgt)
     * TF:
     *     source.weights / target.weights をペアで assign
     */
    public function softUpdate(object $g, AbstractModel $source, AbstractModel $target, float $tau) : void
    {
        $srcVars = $source->trainableVariables();
        $tgtVars = $target->trainableVariables();
        foreach($srcVars as $i => $srcW) {
            $tgtW = $tgtVars[$i];
            $scaledSrc = $g->scale($tau, $srcW);
            $scaledTgt = $g->scale(1.0 - $tau, $tgtW);
            $newVal = $g->add($scaledSrc, $scaledTgt);
            $tgtW->assign($newVal);
        }
    }

    public function alpha() : Variable
    {
        return $this->g->exp($this->logAlpha);
    }

    private function rms(array $values) : float
    {
        $sum = 0.0;
        $count = 0;
        array_walk_recursive($values, function($v) use (&$sum, &$count) {
            $sum += (float)$v * (float)$v;
            $count++;
        });
        return $count ? sqrt($sum / $count) : 0.0;
    }

    private function gradientRmsList(array $grads) : array
    {
        return array_map(fn($v) => $this->rms($v->toArray()), $grads);
    }

    private function range(array $values) : array
    {
        $flat = [];
        array_walk_recursive($values, function($v) use (&$flat) { $flat[] = (float)$v; });
        return [min($flat), max($flat), count($flat) ? array_sum($flat) / count($flat) : 0.0];
    }

    public function diagnostics() : array
    {
        // Keep diagnostics independent of a particular Gym environment.
        // The old implementation used a MountainCar-shaped (B, 2) tensor,
        // which made any environment with another observation size fail.
        $obs = $this->g->Variable($this->la->zeros(
            $this->la->alloc([4, $this->obsDim], dtype:NDArray::float32)
        ));
        $mu = $this->actor->diagnosticMu($obs)->value()->toArray();
        [$muMin, $muMax, $muMean] = $this->range($mu);
        [$lsMin, $lsMax, $lsMean] = $this->range($this->actor->diagnosticLogStd()->value()->toArray());
        [$lpMin, $lpMax, $lpMean] = $this->lastLogPi
            ? $this->range($this->lastLogPi->value()->toArray())
            : [0.0, 0.0, 0.0];
        $sigmaZ = $this->actor->diagnosticSigmaZ();
        [$szMin, $szMax, $szMean] = $sigmaZ
            ? $this->range($sigmaZ->value()->toArray())
            : [0.0, 0.0, 0.0];
        $qDataMean = $this->lastQData ? $this->range($this->lastQData->value()->toArray())[2] : 0.0;
        $qPiMean = $this->lastQPi ? $this->range($this->lastQPi->value()->toArray())[2] : 0.0;
        $targetQMean = $this->lastTargetQ ? $this->range($this->lastTargetQ->value()->toArray())[2] : 0.0;
        return [
            'muMean' => $muMean, 'muMin' => $muMin, 'muMax' => $muMax,
            'logStdMean' => $lsMean, 'logStdMin' => $lsMin, 'logStdMax' => $lsMax,
            'logPiMean' => $lpMean, 'logPiMin' => $lpMin, 'logPiMax' => $lpMax,
            'sigmaZMean' => $szMean, 'sigmaZMin' => $szMin, 'sigmaZMax' => $szMax,
            'qDataMean' => $qDataMean, 'qPiMean' => $qPiMean, 'targetQMean' => $targetQMean,
            'actorGradRms' => $this->rms(array_map(fn($v)=>$v->toArray(), $this->lastActorGrads)),
            'actorGradRmsByVar' => $this->gradientRmsList($this->lastActorGrads),
            'criticGradRms' => $this->rms(array_map(fn($v)=>$v->toArray(), $this->lastCriticGrads)),
        ];
    }


    /**
     * 行動選択
     */
    public function sampleNoise() : Variable
    {
        return $this->actor->sampleNoise();
    }

    public function selectAction(NDArray $obs, Variable $wNoise) : NDArray
    {
        $obsT  = $this->g->Variable($this->la->expandDims($obs, 0));
        $actionVar = $this->actor->forwardInference($obsT, $wNoise);
        $action = $actionVar->value();
        
        $actionFlat = $action->reshape([$this->actDim]);
        $actionSc = $this->la->scal($this->actLimit, $actionFlat);
        
        return $this->clipNdarray($actionSc, -$this->actLimit, $this->actLimit);
    }

    public function selectActionDeterministic(NDArray $obs) : NDArray
    {
        // 評価用: 探索ノイズなしで行動を選ぶ。
        $obsT  = $this->g->Variable($this->la->expandDims($obs, 0));
        $actionVar = $this->actor->forwardDeterministic($obsT);
        $action = $actionVar->value();

        $actionFlat = $action->reshape([$this->actDim]);
        $actionSc = $this->la->scal($this->actLimit, $actionFlat);

        return $this->clipNdarray($actionSc, -$this->actLimit, $this->actLimit);
    }

    /**
     * 学習済みの重みを1つのチェックポイントへ保存する。
     *
     * SACは複数のModelを持つため、各Modelの標準重み形式をまとめて保存する。
     * ActorだけでなくTarget Criticとlog(alpha)も保存するので、同じ構成の
     * エージェントを作成してから loadWeightsFromFile() を呼べば学習を再開できる。
     */
    public function saveWeightsToFile(string $filepath, ?bool $portable = true) : void
    {
        $directory = dirname($filepath);
        if ($directory !== '' && !is_dir($directory)) {
            if (!mkdir($directory, 0777, true) && !is_dir($directory)) {
                throw new \RuntimeException("Could not create checkpoint directory: {$directory}");
            }
        }

        $weights = [];
        $weights['actor'] = [];
        $weights['critic'] = [];
        $weights['criticTarget'] = [];
        $this->actor->saveWeights($weights['actor'], $portable);
        $this->critic->saveWeights($weights['critic'], $portable);
        $this->criticTarget->saveWeights($weights['criticTarget'], $portable);

        $weights['logAlpha'] = $this->logAlpha->value()->toArray();
        $checkpoint = [
            'format' => 'rindow-rl-sac-gsde',
            'version' => self::CHECKPOINT_VERSION,
            'obsDim' => $this->obsDim,
            'actDim' => $this->actDim,
            'weights' => $weights,
        ];

        $data = serialize($checkpoint);
        $temporary = $filepath . '.tmp';
        if (file_put_contents($temporary, $data, LOCK_EX) === false) {
            throw new \RuntimeException("Could not write checkpoint: {$temporary}");
        }
        // Windowsでは既存ファイルへのrenameが失敗することがあるため、
        // 一時ファイルの書き込み成功後にチェックポイントを置き換える。
        if (is_file($filepath) && !unlink($filepath)) {
            @unlink($temporary);
            throw new \RuntimeException("Could not replace checkpoint: {$filepath}");
        }
        if (!rename($temporary, $filepath)) {
            @unlink($temporary);
            throw new \RuntimeException("Could not finalize checkpoint: {$filepath}");
        }
    }

    /**
     * saveWeightsToFile() で保存したチェックポイントを復元する。
     */
    public function loadWeightsFromFile(string $filepath) : void
    {
        if (!is_file($filepath)) {
            throw new \InvalidArgumentException("Checkpoint does not exist: {$filepath}");
        }
        $checkpoint = unserialize(file_get_contents($filepath), ['allowed_classes' => false]);
        if (!is_array($checkpoint)
            || ($checkpoint['format'] ?? null) !== 'rindow-rl-sac-gsde'
            || ($checkpoint['version'] ?? null) !== self::CHECKPOINT_VERSION
        ) {
            throw new \UnexpectedValueException("Invalid SAC gSDE checkpoint: {$filepath}");
        }
        if (($checkpoint['obsDim'] ?? null) !== $this->obsDim
            || ($checkpoint['actDim'] ?? null) !== $this->actDim
        ) {
            throw new \InvalidArgumentException('Checkpoint dimensions do not match this agent.');
        }

        $weights = $checkpoint['weights'] ?? null;
        if (!is_array($weights)
            || !isset($weights['actor'], $weights['critic'], $weights['criticTarget'], $weights['logAlpha'])
        ) {
            throw new \UnexpectedValueException('Checkpoint is missing SAC model weights.');
        }
        $this->actor->loadWeights($weights['actor']);
        $this->critic->loadWeights($weights['critic']);
        $this->criticTarget->loadWeights($weights['criticTarget']);
        $this->logAlpha->assign($this->la->array($weights['logAlpha']));
    }
    
    private function clipNdarray(NDArray $x, float $min, float $max) : NDArray
    {
        $arr = $x->toArray();
        array_walk_recursive($arr, function(&$v) use ($min, $max) {
            $v = max(min($v, $max), $min);
        });
        return $this->la->array($arr);
    }


    /**
     * ** 学習 **********************************
     * 
     *   各ブロックが独立した GradientTape を持つ。
     *
     *   PyTorch → TF 対応:
     *       optimizer.zero_grad()           (不要: TF は毎回新しい tape)
     *       loss.backward()              →  grads = tape.gradient(loss, vars)
     *       optimizer.step()             →  opt.apply_gradients(zip(grads, vars))
     *       with torch.no_grad():        →  tape 外 + tf.stop_gradient()
     *
     */
    public function update(ReplayBuffer $buffer) : array
    {
        $g = $this->g;
        [$obs, $actions, $rewards, $nextObs, $dones] = $buffer->sample($this->batchSize);

        $obsV      = $g->Variable($obs);
        $actionsV  = $g->Variable($actions);
        $rewardsV  = $g->Variable($rewards);
        $nextObsV = $g->Variable($nextObs);
        $donesV    = $g->Variable($dones);

        // ── [A] target_q (勾配不要) ──────────────
        // tape 外で計算 → 自動的に勾配追跡なし
        // tf.stop_gradient で念のため勾配を遮断
        [$nextActions, $nextLogPi] = $this->actor->forwardTrain($nextObsV);
        $nextActionsSc = $g->mul($nextActions, $this->actLimit);
        
        [$q1Next, $q2Next] = $this->criticTarget->forward($nextObsV, $nextActionsSc);
        $qNextMin = $g->minimum($q1Next, $q2Next);
        
        $alphaNextLogPi = $g->mul($this->alpha(), $nextLogPi);
        $qNext = $g->sub($qNextMin, $alphaNextLogPi);
        
        $oneMinusDones = $g->sub(1.0, $donesV);
        $gammaDonesQNext = $g->mul($this->gamma, $g->mul($oneMinusDones, $qNext));
        $targetQ = $g->stopGradient($g->add($rewardsV, $gammaDonesQNext));
        $this->lastTargetQ = $targetQ;

        // ── [B] Critic 更新 ──────────────────────
        // PyTorch: critic_loss.backward(); critic_opt.step()
        $critic = $this->critic;
        $agent = $this;
        $criticLoss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $critic, $obsV, $actionsV, $targetQ, $agent)
        {
            [$q1, $q2] = $critic->forward($obsV, $actionsV);
            $agent->lastQData = $g->minimum($q1, $q2);
            $criticLoss = $g->add(
                $g->reduceMean($g->square($g->sub($q1, $targetQ))),
                $g->reduceMean($g->square($g->sub($q2, $targetQ)))
            );
            return $criticLoss;
        });

        $criticVars = $critic->trainableVariables();
        $criticGrads = $tape->gradient($criticLoss, $criticVars);
        $this->lastCriticGrads = $criticGrads;
        $this->criticOpt->update($criticVars, $criticGrads);
        $this->critic->syncWeightCaches();

        // ── [C] Actor 更新 ───────────────────────
        $actLimit = $this->actLimit;
        $actor = $this->actor;
        $critic = $this->critic;
        $agent = $this;
        [$actorLoss,$logPi] = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $agent, $actor, $obsV, $actLimit, $critic)
        {
            [$newActions, $logPi] = $actor->forwardTrain($obsV);
            $newActionsSc = $g->mul($newActions, $actLimit);
            [$q1Pi, $q2Pi] = $critic->forward($obsV, $newActionsSc);
            $agent->lastQPi = $g->minimum($q1Pi, $q2Pi);
            $actorLoss = $g->reduceMean($g->sub($g->mul($g->stopGradient($agent->alpha()), $logPi), $g->minimum($q1Pi, $q2Pi)));
            return [$actorLoss,$logPi];
        });
        
        $actorVars = $this->actor->trainableVariables();
        $actorGrads = $tape->gradient($actorLoss, $actorVars);
        $this->lastLogPi = $logPi;
        $this->lastActorGrads = $actorGrads;
        $this->actorOpt->update($actorVars, $actorGrads);
        $this->actor->syncWeightCaches();
        if (getenv('RL_FREEZE_LOG_STD') === '1') {
            $this->actor->resetLogStd();
        }

        // ── [D] Alpha 更新 ───────────────────────
        $logAlpha = $this->logAlpha;
        $targetEntropy = $this->targetEntropy;
        $alphaLoss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $logAlpha, $logPi, $targetEntropy)
        {
            $alphaLoss = $g->scale(-1.0, $g->reduceMean($g->mul($logAlpha, $g->stopGradient($g->add($logPi, $targetEntropy)))));
            return $alphaLoss;
        });
        $alphaVars = [$this->logAlpha];
        $alphaGrads = $tape->gradient($alphaLoss, $alphaVars);
        $this->alphaOpt->update($alphaVars, $alphaGrads);

        // ── [E] Critic ソフトアップデート ────────
        $this->softUpdate($this->g, $this->critic, $this->criticTarget, $this->tau);
        $this->criticTarget->syncWeightCaches();

        return [
            "critic_loss" => $criticLoss->value()->toArray(),
            "actor_loss"  => $actorLoss->value()->toArray(),
            "alpha"       => $this->alpha()->value()->toArray(),
        ];
    }
}
