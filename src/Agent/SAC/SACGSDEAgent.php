<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;


/** Soft Actor-Critic agent with generalized state-dependent exploration. */
class SACGSDEAgent
{
    private const CHECKPOINT_VERSION = 1;
    private Builder $nn;
    private object $la;
    private object $backend;
    private object $g;
    private int $actDim;
    /** @var int|array<int,int> */
    private int|array $obsDim;
    private float $actLimit;
    public GSDEActor $actor;
    public Critic $critic;
    public Critic $criticTarget;
    private float $targetEntropy;
    private Variable $logAlpha;
    private object $actorOpt;
    private object $criticOpt;
    private object $alphaOpt;
    /** @var array<int,NDArray> */
    private array $lastActorGrads = [];
    /** @var array<int,NDArray> */
    private array $lastCriticGrads = [];
    private ?Variable $lastLogPi = null;
    private ?Variable $lastQData = null;
    private ?Variable $lastQPi = null;
    private ?Variable $lastTargetQ = null;
    private float $gamma;
    private float $tau;
    private int $batchSize;

    /**
     * @param int|array<int,int> $obsDim
     * @param array<int,object>|null $featureLayers
     */
    public function __construct(
        Builder $nn,
        int|array $obsDim,
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
        ?array $featureLayers=null,
    )
    {
        $this->nn = $nn;
        $this->backend = $nn->backend();
        $this->la = $this->backend->primaryLA();
        $this->g = $nn->gradient();
        if ($featureLayers === []) $featureLayers = null;
        $observationShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        if ($observationShape === []
            || array_filter($observationShape,static fn(int $dim)=>$dim < 1)) {
            throw new \InvalidArgumentException('Invalid SAC observation dimensions.');
        }
        $this->actDim   = $actDim;
        $this->obsDim   = is_int($obsDim) ? $obsDim : $observationShape;
        $this->actLimit = $actLimit;
        $this->gamma = $gamma;
        $this->tau = $tau;
        $this->batchSize = $batchSize;
        $la = $this->la; 

        $this->actor = new GSDEActor(
            $nn,$this->obsDim,$actDim,$gsdeLatentDim,$hiddenDim,$featureLayers
        );
        $this->critic = new Critic(
            $nn,$this->obsDim,$actDim,$hiddenDim,$featureLayers
        );
        $this->criticTarget = new Critic(
            $nn,$this->obsDim,$actDim,$hiddenDim,$featureLayers
        );

        // Build every model before copying parameters into the target critic.
        $batchedObservationShape = array_merge([1],$observationShape);
        $dummyObs = $this->g->Variable($la->zeros(
            $la->alloc($batchedObservationShape,dtype:NDArray::float32)
        ));
        $dummyAct = $this->g->Variable($la->zeros($la->alloc([1, $actDim])));
        
        $this->actor->build($batchedObservationShape);
        $this->critic->build($batchedObservationShape,[1,$actDim]);
        $this->criticTarget->build($batchedObservationShape,[1,$actDim]);
        $this->actor->forwardTrain($dummyObs);
        $this->critic->forward($dummyObs, $dummyAct);
        $this->criticTarget->forward($dummyObs, $dummyAct);
        
        $criticVars = $this->critic->trainableVariables();
        $criticVars = $this->critic->variables();


        $this->softUpdate($this->g, $this->critic, $this->criticTarget, 1.0);  // Exact initial copy.

        $this->actorOpt  = $nn->optimizers->Adam(lr: $lrActor);
        $this->criticOpt = $nn->optimizers->Adam(lr: $lrCritic);
        $this->alphaOpt  = $nn->optimizers->Adam(lr: $lrAlpha);

        // Optimize log(alpha) to keep entropy near the target value.
        // PyTorch: torch.tensor(log(ALPHA_INIT), requires_grad=True)
        // TF:      tf.Variable(..., trainable=True)
        $this->targetEntropy = -(float)$actDim;
        $this->logAlpha = $this->g->Variable(
            $this->la->array([log($alphaInit)]),
            trainable:true, name:"log_alpha"
        );
    }

    public function summary() : void
    {
        echo "***** Actor Network *****\n";
        $this->actor->summary();
        echo "\n";
        echo "***** Critic Network *****\n";
        $this->critic->summary();
    }

    /** Applies target = tau*source + (1-tau)*target in place. */
    public function softUpdate(object $g, AbstractModel $source, AbstractModel $target, float $tau) : void
    {
        $srcVars = $source->trainableVariables();
        $tgtVars = $target->trainableVariables();
        foreach($srcVars as $i => $srcW) {
            $tgtW = $tgtVars[$i];
            $scaledSrc = $g->scale($tau, $srcW);
            $scaledTgt = $g->scale(1.0 - $tau, $tgtW);
            $newVal = $g->add($scaledSrc, $scaledTgt);
            $this->la->copy($newVal->value(),$tgtW->value());
        }
    }

    public function alpha() : Variable
    {
        return $this->g->exp($this->logAlpha);
    }

    /** @return array{NDArray,NDArray,NDArray} */
    private function rangeStatistics(NDArray $values) : array
    {
        $flat = $values->reshape([$values->size()]);
        return [
            $this->la->min($flat),
            $this->la->max($flat),
            $this->la->reduceMean($flat,axis:0),
        ];
    }

    private function rmsStatistic(NDArray $values) : NDArray
    {
        $flat = $values->reshape([$values->size()]);
        return $this->la->sqrt($this->la->reduceMean(
            $this->la->square($this->la->copy($flat)),axis:0
        ));
    }

    /** @param array<NDArray> $gradients */
    private function gradientRms(array $gradients) : NDArray
    {
        if (count($gradients)===0) {
            return $this->la->array(0.0,dtype:NDArray::float32);
        }
        $sumSquares = $this->la->zeros($this->la->alloc(
            [],dtype:$gradients[0]->dtype()
        ));
        $count = 0;
        foreach ($gradients as $gradient) {
            $flat = $gradient->reshape([$gradient->size()]);
            $squares = $this->la->square($this->la->copy($flat));
            $this->la->axpy(
                $this->la->reduceSum($squares,axis:0),$sumSquares
            );
            $count += $gradient->size();
        }
        return $this->la->sqrt($this->la->scal(1.0/$count,$sumSquares));
    }

    /** @return array<string,float|array<int,float>> */
    public function diagnostics() : array
    {
        // Keep diagnostics independent of a particular Gym environment.
        // The old implementation used a MountainCar-shaped (B, 2) tensor,
        // which made any environment with another observation size fail.
        $obs = $this->g->Variable($this->la->zeros(
            $this->la->alloc(array_merge([4],$this->observationShape()), dtype:NDArray::float32)
        ));
        [$muMin, $muMax, $muMean] = $this->rangeStatistics(
            $this->actor->diagnosticMu($obs)->value()
        );
        [$lsMin, $lsMax, $lsMean] = $this->rangeStatistics(
            $this->actor->diagnosticLogStd()->value()
        );
        $zero = $this->la->array(0.0,dtype:NDArray::float32);
        [$lpMin, $lpMax, $lpMean] = $this->lastLogPi
            ? $this->rangeStatistics($this->lastLogPi->value())
            : [$zero, $zero, $zero];
        $sigmaZ = $this->actor->diagnosticSigmaZ();
        [$szMin, $szMax, $szMean] = $sigmaZ
            ? $this->rangeStatistics($sigmaZ->value())
            : [$zero, $zero, $zero];
        $qDataMean = $this->lastQData
            ? $this->rangeStatistics($this->lastQData->value())[2] : $zero;
        $qPiMean = $this->lastQPi
            ? $this->rangeStatistics($this->lastQPi->value())[2] : $zero;
        $targetQMean = $this->lastTargetQ
            ? $this->rangeStatistics($this->lastTargetQ->value())[2] : $zero;

        $metrics = [
            'muMean' => $muMean, 'muMin' => $muMin, 'muMax' => $muMax,
            'logStdMean' => $lsMean, 'logStdMin' => $lsMin, 'logStdMax' => $lsMax,
            'logPiMean' => $lpMean, 'logPiMin' => $lpMin, 'logPiMax' => $lpMax,
            'sigmaZMean' => $szMean, 'sigmaZMin' => $szMin, 'sigmaZMax' => $szMax,
            'qDataMean' => $qDataMean, 'qPiMean' => $qPiMean, 'targetQMean' => $targetQMean,
            'actorGradRms' => $this->gradientRms($this->lastActorGrads),
            'criticGradRms' => $this->gradientRms($this->lastCriticGrads),
        ];
        $actorGradKeys = [];
        foreach ($this->lastActorGrads as $i => $gradient) {
            $key = "actorGradRmsByVar.$i";
            $actorGradKeys[] = $key;
            $metrics[$key] = $this->rmsStatistic($gradient);
        }

        // All diagnostics cross the device boundary in one small vector.
        // min()/max() return PHP scalars on the CPU backend, but zero-dimensional
        // NDArrays on the GPU backend. Normalize both forms before stacking.
        $metricValues = array_map(
            fn($value) => $value instanceof NDArray
                ? $value
                : $this->la->array($value,dtype:NDArray::float32),
            array_values($metrics)
        );
        $values = $this->hostArray(
            $this->la->stack($metricValues)
        )->toArray();
        $diagnostics = array_combine(array_keys($metrics),array_map('floatval',$values));
        $diagnostics['actorGradRmsByVar'] = [];
        foreach ($actorGradKeys as $key) {
            $diagnostics['actorGradRmsByVar'][] = $diagnostics[$key];
            unset($diagnostics[$key]);
        }
        return $diagnostics;
    }


    /** Samples a gSDE noise matrix for caller-managed exploration. */
    public function sampleNoise() : Variable
    {
        return $this->actor->sampleNoise();
    }

    /** @return array<int> */
    public function observationShape() : array
    {
        return is_int($this->obsDim) ? [$this->obsDim] : $this->obsDim;
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
        // Evaluation omits exploration noise.
        $obsT  = $this->g->Variable($this->la->expandDims($obs, 0));
        $actionVar = $this->actor->forwardDeterministic($obsT);
        $action = $actionVar->value();

        $actionFlat = $action->reshape([$this->actDim]);
        $actionSc = $this->la->scal($this->actLimit, $actionFlat);

        return $this->clipNdarray($actionSc, -$this->actLimit, $this->actLimit);
    }

    /**
     * Saves all models and log(alpha) in one checkpoint. The target critic is
     * included so loading the checkpoint can resume training exactly.
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

        $weights['logAlpha'] = $this->hostArray($this->logAlpha->value())->toArray();
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
        // Commit through a temporary file; Windows cannot rename over an existing file.
        if (is_file($filepath) && !unlink($filepath)) {
            @unlink($temporary);
            throw new \RuntimeException("Could not replace checkpoint: {$filepath}");
        }
        if (!rename($temporary, $filepath)) {
            @unlink($temporary);
            throw new \RuntimeException("Could not finalize checkpoint: {$filepath}");
        }
    }

    /** Restores a checkpoint produced by saveWeightsToFile(). */
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
        $this->la->copy(
            $this->la->array(
                $weights['logAlpha'],dtype:$this->logAlpha->value()->dtype()
            ),
            $this->logAlpha->value(),
        );
    }
    
    private function clipNdarray(NDArray $x, float $min, float $max) : NDArray
    {
        return $this->la->minimum(
            $this->la->maximum($this->la->copy($x),$min),$max
        );
    }

    private function hostArray(NDArray $value) : NDArray
    {
        return $this->backend->ndarray($value);
    }

    private function scalar(object $value) : float
    {
        return (float)$this->la->scalar($value->value());
    }

    public function alphaValue() : float
    {
        return $this->scalar($this->alpha());
    }


    /**
     * Updates critic, actor, and entropy coefficient with separate gradient
     * tapes so each optimizer only observes its own objective.
     */
    /** @return array{critic_loss:float,actor_loss:float,alpha:float} */
    public function update(ReplayBuffer $buffer) : array
    {
        $g = $this->g;
        [$obs, $actions, $rewards, $nextObs, $dones] = $buffer->sample($this->batchSize);

        $obsV      = $g->Variable($obs);
        $actionsV  = $g->Variable($actions);
        $rewardsV  = $g->Variable($rewards);
        $nextObsV = $g->Variable($nextObs);
        $notDonesV = $g->sub($g->constant(1.0),$g->Variable($dones));

        // The Bellman target must not propagate gradients into target networks.
        [$nextActions, $nextLogPi] = $this->actor->forwardTrain($nextObsV);
        $nextActionsSc = $g->mul($nextActions, $this->actLimit);
        
        [$q1Next, $q2Next] = $this->criticTarget->forward($nextObsV, $nextActionsSc);
        $qNextMin = $g->minimum($q1Next,$q2Next);
        
        $alphaNextLogPi = $g->mul($this->alpha(), $nextLogPi);
        $qNext = $g->sub($qNextMin, $alphaNextLogPi);
        
        $gammaDonesQNext = $g->mul($this->gamma, $g->mul($notDonesV, $qNext));
        $targetQ = $g->stopGradient($g->add($rewardsV, $gammaDonesQNext));
        $this->lastTargetQ = $targetQ;

        // Fit both Q estimates to the entropy-adjusted Bellman target.
        $critic = $this->critic;
        $agent = $this;
        $criticLoss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $critic, $obsV, $actionsV, $targetQ, $agent)
        {
            [$q1, $q2] = $critic->forward($obsV, $actionsV);
            $agent->lastQData = $g->minimum($q1,$q2);
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

        // Improve the policy against the smaller Q estimate to limit overestimation.
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
            $agent->lastQPi = $g->minimum($q1Pi,$q2Pi);
            $actorLoss = $g->reduceMean($g->sub(
                $g->mul($g->stopGradient($agent->alpha()),$logPi),
                $agent->lastQPi
            ));
            return [$actorLoss,$logPi];
        });
        
        $actorVars = $this->actor->trainableVariables();
        $actorGrads = $tape->gradient($actorLoss, $actorVars);
        $this->lastLogPi = $logPi;
        $this->lastActorGrads = $actorGrads;
        $this->actorOpt->update($actorVars, $actorGrads);
        if (getenv('RL_FREEZE_LOG_STD') === '1') {
            $this->actor->resetLogStd();
        }

        // Adapt the entropy coefficient toward targetEntropy.
        $logAlpha = $this->logAlpha;
        $targetEntropy = $this->targetEntropy;
        $alphaLoss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $logAlpha, $logPi, $targetEntropy)
        {
            $alphaLoss = $g->scale(-1.0, $g->reduceMean($g->mul($logAlpha, $g->stopGradient($g->add($logPi, $g->constant($targetEntropy))))));
            return $alphaLoss;
        });
        $alphaVars = [$this->logAlpha];
        $alphaGrads = $tape->gradient($alphaLoss, $alphaVars);
        $this->alphaOpt->update($alphaVars, $alphaGrads);

        // Move the target critic slowly toward the updated critic.
        $this->softUpdate($this->g, $this->critic, $this->criticTarget, $this->tau);

        return [
            "critic_loss" => $this->scalar($criticLoss),
            "actor_loss"  => $this->scalar($actorLoss),
            "alpha"       => $this->alphaValue(),
        ];
    }
}
