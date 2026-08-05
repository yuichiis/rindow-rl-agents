<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;

/** Synchronous advantage actor-critic for discrete and Gaussian continuous actions. */
class A2CAgent
{
    private const CHECKPOINT_VERSION = 4;
    private object $la;
    private object $g;
    private object $optimizer;
    public ActorCritic $network;

    public function __construct(
        private Builder $nn,
        private int $obsDim,
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
    ) {
        if ($obsDim < 1 || $numActions < ($continuous ? 1 : 2)) {
            throw new \InvalidArgumentException('Invalid observation or action dimension.');
        }
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        if ($continuous && ($actionMin === null || $actionMax === null)) {
            throw new \InvalidArgumentException('Continuous actions require actionMin and actionMax.');
        }
        $this->network = new ActorCritic(
            $nn, $obsDim, $numActions, $hiddenLayers, $continuous, $initialLogStd,
            $actionMin, $actionMax, $actionKernelInitializer, $activation
        );
        $dummy = $this->g->Variable($this->la->zeros($this->la->alloc([1, $obsDim])));
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
    public function observationDimension() : int { return $this->obsDim; }
    public function actionDimension() : int { return $this->numActions; }
    public function isContinuous() : bool { return $this->continuous; }

    public function clipAction(NDArray $action) : NDArray
    {
        if (!$this->continuous) return $action;
        $action = $this->la->copy($action);
        if ($this->actionMin !== null) $action = $this->la->maximum($action, $this->actionMin);
        if ($this->actionMax !== null) $action = $this->la->minimum($action, $this->actionMax);
        return $action;
    }

    /** @return array{int|NDArray,float} sampled action and V(s) */
    public function selectAction(NDArray $observation) : array
    {
        if ($this->continuous) return $this->selectContinuousAction($observation);
        [$probs, $value] = $this->inference($observation);
        $thresholds = $this->la->cumsum($this->la->copy($probs), axis:-1);
        $rand = $this->la->randomUniform([1], dtype:$probs->dtype(), low:0.0, high:1.0);
        $action = (int)$this->la->searchsorted($thresholds, $rand, true)->toArray()[0];
        return [$action, $value];
    }

    public function selectActionDeterministic(NDArray $observation) : int|NDArray
    {
        if ($this->continuous) {
            $batch = $this->asBatch($observation);
            [$mean] = $this->network->forward($this->g->Variable($batch), false);
            return $this->clipAction($this->la->squeeze($mean->value(), axis:0));
        }
        [$probs] = $this->inference($observation);
        $values = $probs[0]->toArray();
        $best = 0;
        foreach ($values as $action => $probability) {
            if ($probability > $values[$best]) $best = $action;
        }
        return $best;
    }

    public function value(NDArray $observation) : float
    {
        [, $value] = $this->inference($observation);
        return $value;
    }

    /** @return array{NDArray,float} */
    private function inference(NDArray $observation) : array
    {
        $batch = $this->asBatch($observation);
        [$logits, $value] = $this->network->forward($this->g->Variable($batch), false);
        return [$this->la->softmax($logits->value()), (float)$value->value()->toArray()[0][0]];
    }

    private function asBatch(NDArray $observation) : NDArray
    {
        if ($this->la->isInt($observation)) {
            $observation = $this->la->astype($observation, dtype:NDArray::float32);
        }
        return $this->la->copy($observation)->reshape([1, $this->obsDim]);
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
            (float)$value->value()->toArray()[0][0],
        ];
    }

    /** @return array{policy_loss:float,value_loss:float,entropy:float} */
    public function update(array $rollout) : array
    {
        [$observations, $actions, $advantages, $returns] = $rollout;
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
            function() use ($g, $network, $observations, $actions, $advantages, $returns) {
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
        $array = $value->value()->toArray();
        while (is_array($array)) $array = reset($array);
        return (float)$array;
    }

    private function clipGradients(array $gradients) : array
    {
        if (is_infinite($this->maxGradNorm)) return $gradients;
        $sumSquares = 0.0;
        foreach ($gradients as $gradient) {
            $stack = [$gradient->toArray()];
            while ($stack !== []) {
                $value = array_pop($stack);
                if (is_array($value)) foreach ($value as $item) $stack[] = $item;
                else $sumSquares += (float)$value * (float)$value;
            }
        }
        $norm = sqrt($sumSquares);
        if ($norm <= $this->maxGradNorm || $norm == 0.0) return $gradients;
        $scale = $this->maxGradNorm / ($norm + 1.0e-8);
        foreach ($gradients as $i => $gradient) $gradients[$i] = $this->la->scal($scale, $gradient);
        return $gradients;
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
