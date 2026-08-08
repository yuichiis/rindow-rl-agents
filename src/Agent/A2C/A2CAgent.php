<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;

/** Synchronous advantage actor-critic for discrete and Gaussian continuous actions. */
class A2CAgent
{
    private const CHECKPOINT_VERSION = 4;
    private object $la;
    private object $backend;
    private object $g;
    private object $optimizer;
    public ActorCritic $network;

    public function __construct(
        private Builder $nn,
        private int|array $obsDim,
        private int $numActions,
        array $hiddenLayers = [64, 64],
        float $learningRate = 7.0e-4,
        private float $valueLossWeight = 0.5,
        private float $entropyWeight = 0.01,
        private float $maxGradNorm = 0.5,
        private bool $normalizeAdvantages = false,
        private bool $continuous = false,
        private ?NDArray $actionMin = null,
        private ?NDArray $actionMax = null,
        float $initialLogStd = -0.5,
        string $optimizer = 'adam',
        mixed $actionKernelInitializer = null,
        string $activation = 'tanh',
        private ?string $stateField = null,
        private ?string $actionMaskField = null,
        // Optional CNN/RNN layers prepended to the shared feature network.
        ?array $featureLayers = null,
    ) {
        if ($featureLayers === []) $featureLayers = null;
        $observationShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        if ($observationShape === []
            || array_filter($observationShape,static fn($dim)=>!is_int($dim) || $dim < 1)
            || $numActions < ($continuous ? 1 : 2)) {
            throw new \InvalidArgumentException('Invalid observation or action dimension.');
        }
        $this->obsDim = is_int($obsDim) ? $obsDim : $observationShape;
        $this->backend = $nn->backend();
        $this->la = $this->backend->primaryLA();
        $this->g = $nn->gradient();
        if ($continuous && ($actionMin === null || $actionMax === null)) {
            throw new \InvalidArgumentException('Continuous actions require actionMin and actionMax.');
        }
        if ($continuous && $actionMaskField !== null) {
            throw new \InvalidArgumentException('Action masks are supported only for discrete actions.');
        }
        $this->network = new ActorCritic(
            $nn, $obsDim, $numActions, $hiddenLayers, $continuous, $initialLogStd,
            $actionMin, $actionMax, $actionKernelInitializer, $activation,
            $featureLayers
        );
        $dummy = $this->g->Variable($this->la->zeros($this->la->alloc(
            array_merge([1],$observationShape),dtype:NDArray::float32
        )));
        $this->network->forward($dummy);
        $this->optimizer = match (strtolower($optimizer)) {
            'adam' => $nn->optimizers->Adam(lr:$learningRate),
            'rmsprop' => $nn->optimizers->RMSprop(
                lr:$learningRate, rho:0.99, epsilon:1.0e-5
            ),
            default => throw new \InvalidArgumentException("Unknown optimizer: {$optimizer}"),
        };
    }

    public function summary() : void { $this->network->summary(); }
    public function observationDimension() : int { return array_product($this->observationShape()); }
    /** @return array<int> */
    public function observationShape() : array
    {
        return is_int($this->obsDim) ? [$this->obsDim] : $this->obsDim;
    }
    public function actionDimension() : int { return $this->numActions; }
    public function isContinuous() : bool { return $this->continuous; }
    public function usesActionMask() : bool { return $this->actionMaskField !== null; }

    /** @return array{NDArray,?NDArray} network state and optional action mask */
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
            throw new \InvalidArgumentException(
                "Observation field '{$this->stateField}' must be an NDArray."
            );
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

    public function clipAction(NDArray $action) : NDArray
    {
        if (!$this->continuous) return $action;
        $action = $this->la->copy($action);
        if ($this->actionMin !== null) $action = $this->la->maximum($action, $this->actionMin);
        if ($this->actionMax !== null) $action = $this->la->minimum($action, $this->actionMax);
        return $action;
    }

    /** @return array{int|NDArray,float} sampled action and V(s) */
    public function selectAction(NDArray|array $observation) : array
    {
        [$observation, $mask] = $this->parseObservation($observation);
        return $this->selectActionFromState($observation, $mask);
    }

    /** @return array{int|NDArray,float} sampled action and V(s) */
    public function selectActionFromState(NDArray $observation, ?NDArray $mask = null) : array
    {
        if ($this->continuous) return $this->selectContinuousAction($observation);
        [$probs, $value] = $this->inference($observation, $mask);
        $thresholds = $this->la->cumsum($this->la->copy($probs), axis:-1);
        $rand = $this->la->randomUniform([1], dtype:$probs->dtype(), low:0.0, high:1.0);
        $selected = $this->la->searchsorted($thresholds, $rand, true);
        $action = (int)$this->hostArray($selected)->toArray()[0];
        return [$action, $value];
    }

    public function selectActionDeterministic(NDArray|array $observation) : int|NDArray
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
            if ($probability > $values[$best]) $best = $action;
        }
        return $best;
    }

    public function value(NDArray|array $observation) : float
    {
        [$observation, $mask] = $this->parseObservation($observation);
        [, $value] = $this->inference($observation, $mask);
        return $value;
    }

    /** @return array{NDArray,float} */
    private function inference(NDArray $observation, ?NDArray $mask = null) : array
    {
        $batch = $this->asBatch($observation);
        [$logits, $value] = $this->network->forward($this->g->Variable($batch), false);
        $logits = $logits->value();
        if ($mask !== null) {
            $batchMask = $this->la->expandDims($mask, axis:0);
            $logits = $this->la->masking($batchMask, $this->la->copy($logits), fill:-1.0e9);
        }
        $hostValue = $this->hostArray($value->value())->toArray();
        return [$this->la->softmax($logits), (float)$hostValue[0][0]];
    }

    private function asBatch(NDArray $observation) : NDArray
    {
        if ($this->la->isInt($observation)) {
            $observation = $this->la->astype($observation, dtype:NDArray::float32);
        }
        return $this->la->copy($observation)->reshape(
            array_merge([1],$this->observationShape())
        );
    }

    /** @return array{NDArray,float} */
    private function selectContinuousAction(NDArray $observation) : array
    {
        [$mean, $value, $logStd] = $this->network->forward(
            $this->g->Variable($this->asBatch($observation)), false
        );
        $mu = $mean->value();
        $ls = $logStd->value();
        $std = $this->la->exp($this->la->copy($ls));
        $noise = $this->la->randomNormal($mu->shape(), 0.0, 1.0);
        $action = $this->la->add($mu, $this->la->multiply($std, $noise));
        return [
            $this->la->squeeze($action, axis:0),
            (float)$this->hostArray($value->value())->toArray()[0][0],
        ];
    }

    /** @return array{policy_loss:float,value_loss:float,entropy:float} */
    public function update(array $rollout) : array
    {
        [$observations, $actions, $advantages, $returns] = $rollout;
        $actionMasks = $rollout[4] ?? null;
        if ($this->normalizeAdvantages && $advantages->shape()[0] > 1) {
            // Keep reductions on the configured LA backend. Besides avoiding
            // transfers, this preserves the successful legacy float32 order.
            $mean = $this->la->reduceMean($advantages, axis:0);
            $centered = $this->la->add(
                $mean, $this->la->copy($advantages), alpha:-1.0
            );
            $variance = $this->la->scal(
                1.0 / $advantages->size(),
                $this->la->reduceSum(
                    $this->la->square($this->la->copy($centered)), axis:0
                )
            );
            $std = $this->la->sqrt($variance);
            $advantages = $this->la->multiply(
                $this->la->reciprocal($std, beta:1.0e-8), $centered
            );
        }
        $g = $this->g;
        $network = $this->network;
        [$totalLoss, $policyLoss, $valueLoss, $entropy, $standardDeviation] = $this->nn->with(
            $tape = $g->GradientTape(),
            function() use ($g, $network, $observations, $actions, $advantages, $returns,
                $actionMasks) {
                $outputs = $network->forward($g->Variable($observations), true);
                [$logits, $values] = $outputs;
                $values = $g->squeeze($values, axis:1);
                $standardDeviation = $g->constant(0.0);
                if ($this->continuous) {
                    // Do not clip logStd. The successful legacy A2C lets the
                    // exploration scale first grow and then shrink naturally.
                    $logStd = $outputs[2];
                    $std = $g->add($g->exp($logStd), $g->constant(1.0e-8));
                    $diff = $g->sub($g->constant($actions), $logits);
                    $z = $g->div($diff, $std);
                    $logProbPerAction = $g->sub(
                        $g->sub(
                            $g->scale(-0.5, $g->square($z)), $g->log($std)
                        ),
                        $g->constant(0.5 * log(2.0 * pi()))
                    );
                    $selectedLogProbs = $g->reduceSum(
                        $logProbPerAction, axis:1, keepdims:true
                    );
                    $selectedLogProbs = $g->reduceSum($selectedLogProbs, axis:-1);
                    $gaussianEntropyConstant = $g->constant(0.5 + 0.5 * log(2.0 * pi()));
                    $entropyPerAction = $g->add($logStd, $gaussianEntropyConstant);
                    $entropyPerAction = $g->add($g->zerosLike($logits), $entropyPerAction);
                    $entropy = $g->reduceMean($g->reduceSum($entropyPerAction, axis:-1));
                    $standardDeviation = $g->reduceMean($std);
                } else {
                    if ($actionMasks !== null) {
                        $logits = $g->masking($actionMasks, $logits, fill:-1.0e9);
                    }
                    $logProbs = $g->logSoftmax($logits);
                    $selectedLogProbs = $g->gather($logProbs, $actions, axis:1, batchDims:1);
                    $probs = $g->softmax($logits);
                    $entropy = $g->scale(-1.0, $g->reduceMean(
                        $g->reduceSum($g->mul($probs, $logProbs), axis:1)
                    ));
                }
                $policyLoss = $g->scale(-1.0, $g->reduceMean(
                    $g->mul($selectedLogProbs, $g->stopGradient($g->constant($advantages)))
                ));
                $valueLoss = $g->reduceMean($g->square(
                    $g->sub($values, $g->constant($returns))
                ));
                $totalLoss = $g->sub(
                    $g->add($policyLoss, $g->scale($this->valueLossWeight, $valueLoss)),
                    $g->scale($this->entropyWeight, $entropy)
                );
                return [$totalLoss, $policyLoss, $valueLoss, $entropy, $standardDeviation];
            }
        );
        $variables = $network->trainableVariables();
        $gradients = $this->clipGradients($tape->gradient($totalLoss, $variables));
        $this->optimizer->update($variables, $gradients);
        $network->syncWeightCaches();
        return ['policy_loss'=>$this->scalar($policyLoss), 'value_loss'=>$this->scalar($valueLoss),
            'entropy'=>$this->scalar($entropy), 'std'=>$this->scalar($standardDeviation)];
    }

    private function scalar(object $value) : float
    {
        return (float)$this->la->scalar($value->value());
    }

    private function clipGradients(array $gradients) : array
    {
        if (is_infinite($this->maxGradNorm)) return $gradients;
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
        if ($norm <= $this->maxGradNorm || $norm == 0.0) return $gradients;
        $scale = $this->maxGradNorm / ($norm + 1.0e-8);
        foreach ($gradients as $i => $gradient) $gradients[$i] = $this->la->scal($scale, $gradient);
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
        $checkpoint = ['format'=>'rindow-rl-a2c', 'version'=>self::CHECKPOINT_VERSION,
            'obsDim'=>$this->obsDim, 'numActions'=>$this->numActions,
            'continuous'=>$this->continuous, 'weights'=>$weights];
        if (file_put_contents($filepath, serialize($checkpoint), LOCK_EX) === false) {
            throw new \RuntimeException("Could not write checkpoint: {$filepath}");
        }
    }

    public function loadWeightsFromFile(string $filepath) : void
    {
        $data = file_get_contents($filepath);
        if ($data === false) throw new \RuntimeException("Could not read checkpoint: {$filepath}");
        $checkpoint = unserialize($data, ['allowed_classes'=>false]);
        $version = $checkpoint['version'] ?? null;
        $compatibleVersion = $version === self::CHECKPOINT_VERSION
            || (in_array($version, [1, 2, 3], true) && !$this->continuous);
        if (!is_array($checkpoint) || ($checkpoint['format'] ?? null) !== 'rindow-rl-a2c'
            || !$compatibleVersion
            || ($checkpoint['obsDim'] ?? null) !== $this->obsDim
            || ($checkpoint['numActions'] ?? null) !== $this->numActions
            || ($checkpoint['continuous'] ?? false) !== $this->continuous) {
            throw new \UnexpectedValueException("Invalid or incompatible A2C checkpoint: {$filepath}");
        }
        $this->network->loadWeights($checkpoint['weights']);
    }
}
