<?php
namespace Rindow\RL\Agents\Agent\PPO;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;

/** 離散行動とGaussian/gSDE連続行動に対応するPPOエージェント。 */
class PPOAgent
{
    private const CHECKPOINT_VERSION = 3;
    private object $la;
    private object $backend;
    private object $g;
    private object $optimizer;
    public ActorCritic $network;
    private ?NDArray $sdeNoise = null;

    public function __construct(
        private Builder $nn,
        private int|array $obsDim,
        private int $numActions,
        array $hiddenLayers = [64, 64],
        private float $learningRate = 3.0e-4,
        private float $clipRange = 0.2,
        private float $valueLossWeight = 0.5,
        private float $entropyWeight = 0.01,
        private int $epochs = 10,
        private int $batchSize = 64,
        private float $maxGradNorm = 0.5,
        private bool $clipValueLoss = true,
        private bool $sharedBackbone = false,
        private bool $continuous = false,
        private ?NDArray $actionMin = null,
        private ?NDArray $actionMax = null,
        private string $exploration = 'gaussian',
        private int $sdeSampleFreq = -1,
        private float $sdeInitialLogStd = -2.0,
        private ?string $stateField = null,
        private ?string $actionMaskField = null,
        // Optional CNN/RNN feature extractor used by the shared backbone.
        ?array $featureLayers = null,
    ) {
        if ($featureLayers === []) $featureLayers = null;
        $observationShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        if ($observationShape === []
            || array_filter($observationShape,static fn($dim)=>!is_int($dim) || $dim < 1)) {
            throw new \InvalidArgumentException('Invalid PPO observation dimensions.');
        }
        if ($featureLayers !== null && !$sharedBackbone) {
            throw new \InvalidArgumentException('featureLayers requires sharedBackbone:true.');
        }
        if (!in_array($exploration, ['gaussian', 'gsde'], true)) {
            throw new \InvalidArgumentException("exploration must be 'gaussian' or 'gsde'.");
        }
        if ($exploration === 'gsde' && (!$continuous || !$sharedBackbone)) {
            throw new \InvalidArgumentException('gSDE requires continuous:true and sharedBackbone:true.');
        }
        if ($continuous && $actionMaskField !== null) {
            throw new \InvalidArgumentException('Action masks are supported only for discrete actions.');
        }
        $this->obsDim = is_int($obsDim) ? $obsDim : $observationShape;
        $this->backend = $nn->backend();
        $this->la = $this->backend->primaryLA();
        $this->g = $nn->gradient();
        $this->network = new ActorCritic(
            $nn, $obsDim, $numActions, $hiddenLayers, $sharedBackbone, $continuous,
            $exploration === 'gsde', $sdeInitialLogStd, $featureLayers
        );
        $dummy = $this->g->Variable($this->la->zeros(
            $this->la->alloc(array_merge([1],$observationShape),dtype:NDArray::float32)
        ));
        $this->network->forward($dummy);
        $this->optimizer = $nn->optimizers->Adam(lr:$learningRate, epsilon:1.0e-8);
    }

    public function summary() : void
    {
        $this->network->summary();
    }

    public function isContinuous() : bool { return $this->continuous; }
    public function observationDimension() : int { return array_product($this->observationShape()); }
    /** @return array<int> */
    public function observationShape() : array
    {
        return is_int($this->obsDim) ? [$this->obsDim] : $this->obsDim;
    }
    public function usesActionMask() : bool { return $this->actionMaskField !== null; }
    public function usesSDE() : bool { return $this->exploration === 'gsde'; }
    public function sdeSampleFreq() : int { return $this->sdeSampleFreq; }

    public function resetExplorationNoise() : void
    {
        if ($this->usesSDE()) $this->sdeNoise = $this->network->sampleSDENoise();
    }

    public function clipAction(NDArray $action) : NDArray
    {
        if (!$this->continuous) return $action;
        $out = $this->la->copy($action);
        if ($this->actionMin !== null) $out = $this->la->maximum($out, $this->actionMin);
        if ($this->actionMax !== null) $out = $this->la->minimum($out, $this->actionMax);
        return $out;
    }

    /** @return array{NDArray,?NDArray} state and action mask */
    public function parseObservation(NDArray|array $observation) : array
    {
        if ($observation instanceof NDArray) {
            if ($this->stateField !== null || $this->actionMaskField !== null) {
                throw new \InvalidArgumentException('A dictionary observation was expected.');
            }
            return [$this->asNetworkState($observation), null];
        }
        if ($this->stateField === null) {
            throw new \InvalidArgumentException('stateField is required for dictionary observations.');
        }
        $state = $observation[$this->stateField] ?? null;
        if (!$state instanceof NDArray) {
            throw new \InvalidArgumentException("Observation field '{$this->stateField}' must be an NDArray.");
        }
        $mask = null;
        if ($this->actionMaskField !== null) {
            $mask = $observation[$this->actionMaskField] ?? null;
            if (!$mask instanceof NDArray) {
                throw new \InvalidArgumentException(
                    "Observation field '{$this->actionMaskField}' must be an NDArray."
                );
            }
            if ($mask->shape() !== [$this->numActions]) {
                throw new \InvalidArgumentException('Action mask shape must equal [numActions].');
            }
            if ($mask->dtype() !== NDArray::bool) {
                $mask = $this->la->astype($mask, dtype:NDArray::bool);
            }
            if (!in_array(true, $this->hostArray($mask)->toArray(), true)) {
                throw new \InvalidArgumentException('Action mask must allow at least one action.');
            }
        }
        return [$this->asNetworkState($state), $mask];
    }

    private function asNetworkState(NDArray $state) : NDArray
    {
        if ($state->shape() !== $this->observationShape()) {
            throw new \InvalidArgumentException(sprintf(
                'Observation shape must be [%s]; [%s] given.',
                implode(',',$this->observationShape()),implode(',',$state->shape())
            ));
        }
        return $this->la->isInt($state)
            ? $this->la->astype($state, dtype:NDArray::float32)
            : $state;
    }

    /** @return array{mixed,float,float} action, log probability, value */
    public function selectAction(NDArray|array $observation) : array
    {
        [$observation, $mask] = $this->parseObservation($observation);
        return $this->selectActionFromState($observation, $mask);
    }

    /** @return array{mixed,float,float} action, log probability, value */
    public function selectActionFromState(NDArray $observation, ?NDArray $mask = null) : array
    {
        if ($this->continuous) return $this->selectContinuousAction($observation);
        [$probs, $value] = $this->inference($observation, $mask);
        // Match the legacy Random::randomCategorical implementation exactly:
        // it draws a LA randomUniform value, then applies cumsum/searchsorted.
        // Calling the newer convenience randomCategorical() can use a
        // different backend path and consume a different RNG sequence.
        $thresholds = $this->la->cumsum($this->la->copy($probs), axis:-1);
        $rand = $this->la->randomUniform(
            [1], dtype:$probs->dtype(), low:0.0, high:1.0
        );
        $selected = $this->la->searchsorted($thresholds, $rand, true);
        $action = (int)$this->hostArray($selected)->toArray()[0];
        $hostProbs = $this->hostArray($probs)->toArray();
        $p = max(1.0e-8, (float)$hostProbs[0][$action]);
        return [$action, log($p), $value];
    }

    private function selectContinuousAction(NDArray $observation) : array
    {
        $batch = $this->asBatch($observation);
        if ($this->usesSDE()) {
            if ($this->sdeNoise === null) $this->resetExplorationNoise();
            [$sample, $value, $std, $mean] = $this->network->forwardSDE(
                $this->g->Variable($batch), $this->sdeNoise, false
            );
            $action = $sample->value();
            $ls = $this->la->log($this->la->maximum($std->value(), 1.0e-8));
            // The sampled action is generated by the fixed exploration matrix;
            // its PPO likelihood is the marginal state-dependent Gaussian.
            $logp = $this->gaussianLogProb($action, $mean->value(), $ls);
            return [$this->la->squeeze($action, axis:0), $logp,
                (float)$this->hostArray($value->value())->toArray()[0][0]];
        }
        [$mean, $value, $logStd] = $this->network->forward($this->g->Variable($batch), false);
        $mu = $mean->value();
        $ls = $this->la->minimum($this->la->maximum($logStd, -5.0), 2.0);
        $noise = $this->la->randomNormal($mu->shape(), 0.0, 1.0);
        $std = $this->la->exp($this->la->copy($ls));
        $action = $this->la->add($mu, $this->la->multiply($std, $noise));
        $logp = $this->gaussianLogProb($action, $mu, $ls);
        return [$this->la->squeeze($action, axis:0), $logp,
            (float)$this->hostArray($value->value())->toArray()[0][0]];
    }

    public function selectActionDeterministic(NDArray|array $observation) : mixed
    {
        [$observation, $mask] = $this->parseObservation($observation);
        if ($this->continuous) {
            $batch = $this->asBatch($observation);
            [$mean] = $this->network->forward($this->g->Variable($batch), false);
            return $this->clipAction($this->la->squeeze($mean->value(), axis:0));
        }
        [$probs] = $this->inference($observation, $mask);
        $values = $this->hostArray($probs)[0]->toArray();
        $best = 0;
        foreach ($values as $action => $probability) {
            if ($probability > $values[$best]) {
                $best = $action;
            }
        }
        return $best;
    }

    private function gaussianLogProb(mixed $actions, mixed $mean, mixed $logStd) : mixed
    {
        if ($actions instanceof NDArray && $mean instanceof NDArray && $logStd instanceof NDArray) {
            $diff = $this->la->axpy($mean, $this->la->copy($actions), -1.0);
            $std = $this->la->exp($this->la->copy($logStd));
            $z = $this->la->multiply($diff, $this->la->reciprocal($std));
            $term = $this->la->add(
                $this->la->scal(-0.5, $this->la->square($z)),
                $this->la->scal(-1.0, $logStd)
            );
            $value = $this->hostArray(
                $this->la->reduceSum($term, axis:1)
            )->toArray();
            while (is_array($value)) $value = reset($value);
            return (float)$value;
        }
        $g = $this->g;
        $actionConst = $g->constant($this->la->copy($actions));
        $diff = $g->add($mean, $g->scale(-1.0, $actionConst));
        return $g->squeeze($g->scale(-0.5, $g->square($diff)), axis:1);
    }

    public function value(NDArray|array $observation) : float
    {
        [$observation, $mask] = $this->parseObservation($observation);
        [, $value] = $this->inference($observation, $mask);
        return $value;
    }

    private function inference(NDArray $observation, ?NDArray $mask = null) : array
    {
        $batch = $this->asBatch($observation);
        $obsV = $this->g->Variable($batch);
        [$logits, $value] = $this->network->forward($obsV, false);
        $logits = $logits->value();
        if ($mask !== null) {
            $batchMask = $this->la->expandDims($mask, axis:0);
            $logits = $this->la->masking($batchMask, $this->la->copy($logits), fill:-1.0e9);
        }
        $probs = $this->la->softmax($logits);
        return [$probs,
            (float)$this->hostArray($value->value())->toArray()[0][0]];
    }

    private function asBatch(NDArray $observation) : NDArray
    {
        $batch = $this->la->copy($observation)->reshape(
            array_merge([1],$this->observationShape())
        );
        return $this->la->isInt($batch)
            ? $this->la->astype($batch,dtype:NDArray::float32)
            : $batch;
    }

    /** @return array{policy_loss:float,value_loss:float,entropy:float} */
    public function update(array $rollout) : array
    {
        [$observations, $actions, $oldLogProbs, $advantages, $returns, $oldValues] = $rollout;
        $actionMasks = $rollout[6] ?? null;
        $count = $observations->shape()[0];

        $mean = $this->la->reduceMean($advantages,axis:0);
        $centered = $this->la->add(
            $mean,$this->la->copy($advantages),alpha:-1.0
        );
        $variance = $this->la->scal(
            1.0 / max(1,$count),
            $this->la->reduceSum(
                $this->la->square($this->la->copy($centered)),axis:0
            )
        );
        $std = $this->la->sqrt($variance);
        $advantages = $this->la->multiply(
            $this->la->reciprocal($std,beta:1.0e-8),$centered
        );

        $policyTotal = $valueTotal = $entropyTotal = 0.0;
        $updates = 0;
        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            // Match NDArrayDataset's shuffle semantics used by the old PPO:
            // shuffle batch blocks first, then shuffle items inside each block.
            // PHP's shuffle() uses a different RNG stream and breaks seeded
            // reproducibility even when the LA seed is identical.
            $batchCount = (int)ceil($count / $this->batchSize);
            $batchOrder = $batchCount > 1
                ? $this->hostArray($this->la->randomSequence($batchCount))->toArray()
                : null;
            for ($batchNo = 0; $batchNo < $batchCount; $batchNo++) {
                $block = $batchOrder === null ? 0 : (int)$batchOrder[$batchNo];
                $start = $block * $this->batchSize;
                $size = min($this->batchSize, $count - $start);
                if ($size > 1) {
                    $itemOrder = $this->hostArray(
                        $this->la->randomSequence($size)
                    )->toArray();
                    $batchIndices = [];
                    for ($i = 0; $i < $size; $i++) {
                        $batchIndices[] = $start + (int)$itemOrder[$i];
                    }
                } else {
                    $batchIndices = [$start];
                }
                $idx = $this->la->array($batchIndices, dtype:NDArray::int32);
                $obs = $this->la->gather($observations, $idx);
                $act = $this->la->gather($actions, $idx);
                $oldLog = $this->la->gather($oldLogProbs, $idx);
                $adv = $this->la->gather($advantages, $idx);
                $ret = $this->la->gather($returns, $idx);
                $oldValue = $this->la->gather($oldValues, $idx);
                $mask = $actionMasks === null ? null : $this->la->gather($actionMasks, $idx);
                [$policyLoss, $valueLoss, $entropy] = $this->updateBatch(
                    $obs, $act, $oldLog, $oldValue, $adv, $ret, $mask
                );
                $policyTotal += $policyLoss;
                $valueTotal += $valueLoss;
                $entropyTotal += $entropy;
                $updates++;
            }
        }
        return [
            'policy_loss' => $policyTotal / $updates,
            'value_loss' => $valueTotal / $updates,
            'entropy' => $entropyTotal / $updates,
        ];
    }

    private function updateBatch(
        NDArray $obs,
        NDArray $actions,
        NDArray $oldLog,
        NDArray $oldValues,
        NDArray $advantages,
        NDArray $returns,
        ?NDArray $actionMasks = null,
    ) : array
    {
        $g = $this->g;
        $network = $this->network;
        $clipRange = $this->clipRange;
        $entropyWeight = $this->entropyWeight;
        $valueLossWeight = $this->valueLossWeight;
        $clipValueLoss = $this->clipValueLoss;
        $continuous = $this->continuous;
        $useSDE = $this->usesSDE();
        [$totalLoss, $policyLoss, $valueLoss, $entropy] = $this->nn->with(
            $tape = $g->GradientTape(), function() use (
            $g, $network, $obs, $actions, $oldLog, $oldValues, $advantages, $returns,
            $clipRange, $entropyWeight, $valueLossWeight, $clipValueLoss, $continuous, $useSDE,
            $actionMasks
        ) {
            $outputs = $network->forward($g->Variable($obs), true);
            [$logits, $values] = $outputs;
            $values = $g->squeeze($values, axis:1);
            if ($continuous) {
                // Bound exploration scale to keep exp(logStd) finite.
                $logStd = $useSDE
                    ? $g->log($g->maximum($outputs[2], $g->constant(1.0e-8)))
                    : $g->clipByValue($outputs[2], -5.0, 2.0);
                $stableStd = $useSDE ? $outputs[2] : $g->add($g->exp($logStd), $g->constant(1.0e-8));
                $actionConstant = $g->constant($this->la->copy($actions));
                $diff = $g->add($logits, $g->scale(-1.0, $actionConstant));
                $z = $g->div($diff, $stableStd);
                $selected = $g->reduceSum(
                    $g->sub(
                        $g->scale(-0.5, $g->square($z)),
                        $g->log($stableStd)
                    ), axis:1
                );
                $logProbs = null;
            } else {
                if ($actionMasks !== null) {
                    $logits = $g->masking($actionMasks, $logits, fill:-1.0e9);
                }
                $logProbs = $g->logSoftmax($logits);
                $selected = $g->gather($logProbs, $actions, axis:1, batchDims:1);
            }
            $ratio = $g->exp($g->sub($selected, $oldLog));
            $unclipped = $g->mul($ratio, $advantages);
            $clipped = $g->mul($g->clipByValue($ratio, 1.0 - $clipRange, 1.0 + $clipRange), $advantages);
            $surrogate = $g->minimum($unclipped,$clipped);
            $policyLoss = $g->scale(-1.0, $g->reduceMean($surrogate));

            $valueLossUnclipped = $g->square($g->sub($values, $returns));
            if ($clipValueLoss) {
                $valuesClipped = $g->add(
                    $oldValues,
                    $g->clipByValue($g->sub($values, $oldValues), -$clipRange, $clipRange)
                );
                $valueLossClipped = $g->square($g->sub($valuesClipped, $returns));
                $maximumValueLoss = $g->maximum(
                    $valueLossUnclipped,$valueLossClipped
                );
                $valueLoss = $g->scale(
                    0.5,
                    $g->reduceMean($maximumValueLoss)
                );
            } else {
                $valueLoss = $g->reduceMean($valueLossUnclipped);
            }

            if ($continuous) {
                $entropy = $g->reduceMean($g->reduceSum($logStd, axis:1));
            } else {
                $probs = $g->softmax($logits);
                $entropy = $g->scale(-1.0, $g->reduceMean($g->reduceSum($g->mul($probs, $logProbs), axis:1)));
            }
            $totalLoss = $g->sub(
                $g->add($policyLoss, $g->scale($valueLossWeight, $valueLoss)),
                $g->scale($entropyWeight, $entropy)
            );
            return [$totalLoss, $policyLoss, $valueLoss, $entropy];
        });
        $variables = $network->trainableVariables();
        $gradients = $this->clipGradients($tape->gradient($totalLoss, $variables));
        $this->optimizer->update($variables, $gradients);
        $network->syncWeightCaches();
        return [$this->scalar($policyLoss), $this->scalar($valueLoss), $this->scalar($entropy)];
    }

    private function scalar(object $value) : float
    {
        return (float)$this->la->scalar($value->value());
    }

    private function clipGradients(array $gradients) : array
    {
        $sumSquares = 0.0;
        foreach ($gradients as $gradient) {
            $gradientNorm = $this->la->nrm2($gradient);
            if ($gradientNorm instanceof NDArray) {
                $gradientNorm = $this->hostArray($gradientNorm)->toArray();
            }
            $gradientNorm = (float)$gradientNorm;
            $sumSquares += $gradientNorm * $gradientNorm;
        }
        $norm = sqrt($sumSquares);
        if ($norm <= $this->maxGradNorm || $norm == 0.0) {
            return $gradients;
        }
        $scale = $this->maxGradNorm / ($norm + 1.0e-8);
        foreach ($gradients as $i => $gradient) {
            $gradients[$i] = $this->la->scal($scale, $gradient);
        }
        return $gradients;
    }

    private function hostArray(NDArray $value) : NDArray
    {
        return $this->backend->ndarray($value);
    }

    public function saveWeightsToFile(string $filepath, ?bool $portable = true) : void
    {
        $directory = dirname($filepath);
        if (!is_dir($directory) && !mkdir($directory, 0777, true) && !is_dir($directory)) {
            throw new \RuntimeException("Could not create checkpoint directory: {$directory}");
        }
        $weights = [];
        $this->network->saveWeights($weights, $portable);
        $checkpoint = ['format'=>'rindow-rl-ppo', 'version'=>self::CHECKPOINT_VERSION,
            'obsDim'=>$this->obsDim, 'numActions'=>$this->numActions,
            'sharedBackbone'=>$this->sharedBackbone, 'exploration'=>$this->exploration, 'weights'=>$weights];
        if (file_put_contents($filepath, serialize($checkpoint), LOCK_EX) === false) {
            throw new \RuntimeException("Could not write checkpoint: {$filepath}");
        }
    }

    public function loadWeightsFromFile(string $filepath) : void
    {
        $checkpoint = unserialize(file_get_contents($filepath), ['allowed_classes'=>false]);
        $version = $checkpoint['version'] ?? null;
        $compatibleVersion = $version === self::CHECKPOINT_VERSION || $version === 2
            || ($version === 1 && !$this->sharedBackbone && $this->exploration === 'gaussian');
        if (!is_array($checkpoint) || ($checkpoint['format'] ?? null) !== 'rindow-rl-ppo'
            || !$compatibleVersion
            || ($checkpoint['obsDim'] ?? null) !== $this->obsDim
            || ($checkpoint['numActions'] ?? null) !== $this->numActions
            || ($version >= 2 && ($checkpoint['sharedBackbone'] ?? false) !== $this->sharedBackbone)
            || ($version === self::CHECKPOINT_VERSION
                && ($checkpoint['exploration'] ?? 'gaussian') !== $this->exploration)
            || ($version === 2 && $this->exploration !== 'gaussian')) {
            throw new \UnexpectedValueException("Invalid or incompatible PPO checkpoint: {$filepath}");
        }
        $this->network->loadWeights($checkpoint['weights']);
    }
}
