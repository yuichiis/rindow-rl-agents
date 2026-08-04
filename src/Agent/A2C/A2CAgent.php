<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;

/** Synchronous advantage actor-critic agent for discrete action spaces. */
class A2CAgent
{
    private const CHECKPOINT_VERSION = 1;
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
    ) {
        if ($obsDim < 1 || $numActions < 2) {
            throw new \InvalidArgumentException('Invalid observation or action dimension.');
        }
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        $this->network = new ActorCritic($nn, $obsDim, $numActions, $hiddenLayers);
        $dummy = $this->g->Variable($this->la->zeros($this->la->alloc([1, $obsDim])));
        $this->network->forward($dummy);
        $this->optimizer = $nn->optimizers->Adam(lr:$learningRate, epsilon:1.0e-8);
    }

    public function summary() : void { $this->network->summary(); }
    public function observationDimension() : int { return $this->obsDim; }

    /** @return array{int,float} sampled action and V(s) */
    public function selectAction(NDArray $observation) : array
    {
        [$probs, $value] = $this->inference($observation);
        $thresholds = $this->la->cumsum($this->la->copy($probs), axis:-1);
        $rand = $this->la->randomUniform([1], dtype:$probs->dtype(), low:0.0, high:1.0);
        $action = (int)$this->la->searchsorted($thresholds, $rand, true)->toArray()[0];
        return [$action, $value];
    }

    public function selectActionDeterministic(NDArray $observation) : int
    {
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
        if ($this->la->isInt($observation)) {
            $observation = $this->la->astype($observation, dtype:NDArray::float32);
        }
        $batch = $this->la->copy($observation)->reshape([1, $this->obsDim]);
        [$logits, $value] = $this->network->forward($this->g->Variable($batch), false);
        return [$this->la->softmax($logits->value()), (float)$value->value()->toArray()[0][0]];
    }

    /** @return array{policy_loss:float,value_loss:float,entropy:float} */
    public function update(array $rollout) : array
    {
        [$observations, $actions, $advantages, $returns] = $rollout;
        if ($this->normalizeAdvantages && $advantages->shape()[0] > 1) {
            $array = $advantages->toArray();
            $mean = array_sum($array) / count($array);
            $variance = 0.0;
            foreach ($array as $value) $variance += ($value - $mean) ** 2;
            $std = sqrt($variance / count($array) + 1.0e-8);
            foreach ($array as &$value) $value = ($value - $mean) / $std;
            unset($value);
            $advantages = $this->la->array($array, dtype:NDArray::float32);
        }
        $g = $this->g;
        $network = $this->network;
        [$totalLoss, $policyLoss, $valueLoss, $entropy] = $this->nn->with(
            $tape = $g->GradientTape(),
            function() use ($g, $network, $observations, $actions, $advantages, $returns) {
                [$logits, $values] = $network->forward($g->Variable($observations), true);
                $values = $g->squeeze($values, axis:1);
                $logProbs = $g->logSoftmax($logits);
                $selectedLogProbs = $g->gather($logProbs, $actions, axis:1, batchDims:1);
                $policyLoss = $g->scale(-1.0, $g->reduceMean(
                    $g->mul($selectedLogProbs, $g->stopGradient($g->constant($advantages)))
                ));
                $valueLoss = $g->scale(0.5, $g->reduceMean($g->square(
                    $g->sub($values, $g->constant($returns))
                )));
                $probs = $g->softmax($logits);
                $entropy = $g->scale(-1.0, $g->reduceMean(
                    $g->reduceSum($g->mul($probs, $logProbs), axis:1)
                ));
                $totalLoss = $g->sub(
                    $g->add($policyLoss, $g->scale($this->valueLossWeight, $valueLoss)),
                    $g->scale($this->entropyWeight, $entropy)
                );
                return [$totalLoss, $policyLoss, $valueLoss, $entropy];
            }
        );
        $variables = $network->trainableVariables();
        $gradients = $this->clipGradients($tape->gradient($totalLoss, $variables));
        $this->optimizer->update($variables, $gradients);
        $network->syncWeightCaches();
        return ['policy_loss'=>$this->scalar($policyLoss), 'value_loss'=>$this->scalar($valueLoss),
            'entropy'=>$this->scalar($entropy)];
    }

    private function scalar(object $value) : float
    {
        $array = $value->value()->toArray();
        while (is_array($array)) $array = reset($array);
        return (float)$array;
    }

    private function clipGradients(array $gradients) : array
    {
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
            'obsDim'=>$this->obsDim, 'numActions'=>$this->numActions, 'weights'=>$weights];
        if (file_put_contents($filepath, serialize($checkpoint), LOCK_EX) === false) {
            throw new \RuntimeException("Could not write checkpoint: {$filepath}");
        }
    }

    public function loadWeightsFromFile(string $filepath) : void
    {
        $data = file_get_contents($filepath);
        if ($data === false) throw new \RuntimeException("Could not read checkpoint: {$filepath}");
        $checkpoint = unserialize($data, ['allowed_classes'=>false]);
        if (!is_array($checkpoint) || ($checkpoint['format'] ?? null) !== 'rindow-rl-a2c'
            || ($checkpoint['version'] ?? null) !== self::CHECKPOINT_VERSION
            || ($checkpoint['obsDim'] ?? null) !== $this->obsDim
            || ($checkpoint['numActions'] ?? null) !== $this->numActions) {
            throw new \UnexpectedValueException("Invalid or incompatible A2C checkpoint: {$filepath}");
        }
        $this->network->loadWeights($checkpoint['weights']);
    }
}
