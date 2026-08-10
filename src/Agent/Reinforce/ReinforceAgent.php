<?php
namespace Rindow\RL\Agents\Agent\Reinforce;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\RL\Agents\Util\GradientClipping;

/** Monte Carlo policy-gradient agent for discrete action spaces. */
class ReinforceAgent
{
    private const CHECKPOINT_VERSION = 1;
    private object $la;
    private object $backend;
    private object $g;
    private object $optimizer;
    public PolicyNetwork $network;

    public function __construct(
        private Builder $nn,
        private int $obsDim,
        private int $numActions,
        array $hiddenLayers = [128],
        float $learningRate = 1.0e-2,
        private float $entropyWeight = 0.0,
        private float $maxGradNorm = 1.0,
        string $activation = 'relu',
    ) {
        if ($obsDim < 1 || $numActions < 2 || $learningRate <= 0.0) {
            throw new \InvalidArgumentException('Invalid REINFORCE dimensions or learning rate.');
        }
        $this->backend = $nn->backend();
        $this->la = $this->backend->primaryLA();
        $this->g = $nn->gradient();
        $this->network = new PolicyNetwork(
            $nn, $obsDim, $numActions, $hiddenLayers, $activation
        );
        $dummy = $this->g->Variable($this->la->zeros(
            $this->la->alloc([1, $obsDim], dtype:NDArray::float32)
        ));
        $this->network->forward($dummy);
        $this->optimizer = $nn->optimizers->Adam(lr:$learningRate);
    }

    public function summary() : void { $this->network->summary(); }
    public function observationDimension() : int { return $this->obsDim; }
    public function actionDimension() : int { return $this->numActions; }

    public function selectAction(NDArray $observation) : int
    {
        $probs = $this->probabilities($observation);
        $selected = $this->la->randomCategorical($probs);
        return (int)$this->la->scalar($selected)[0];
    }

    public function selectActionDeterministic(NDArray $observation) : int
    {
        $probabilities = $this->probabilities($observation);
        $best = $this->la->reduceArgMax($probabilities,axis:1);
        return (int)$this->la->scalar($best)[0];
    }

    private function probabilities(NDArray $observation) : NDArray
    {
        if ($this->la->isInt($observation)) {
            $observation = $this->la->astype($observation, dtype:NDArray::float32);
        }
        $batch = $this->la->copy($observation)->reshape([1, $this->obsDim]);
        $logits = $this->network->forward($this->g->Variable($batch), false);
        return $this->la->softmax($logits->value());
    }

    /** @return array{policy_loss:float,entropy:float} */
    public function update(NDArray $observations, NDArray $actions, NDArray $returns) : array
    {
        if ($observations->shape()[0] < 1
            || $actions->shape()[0] !== $observations->shape()[0]
            || $returns->shape()[0] !== $observations->shape()[0]) {
            throw new \InvalidArgumentException('Episode arrays must have the same non-zero length.');
        }
        $g = $this->g;
        $network = $this->network;
        [$totalLoss, $policyLoss, $entropy] = $this->nn->with(
            $tape = $g->GradientTape(),
            function() use ($g, $network, $observations, $actions, $returns) {
                $logits = $network->forward($g->Variable($observations), true);
                $logProbs = $g->logSoftmax($logits);
                $selected = $g->gather($logProbs, $actions, axis:1, batchDims:1);
                $weights = $g->stopGradient($g->constant($returns));
                $policyLoss = $g->scale(
                    -1.0, $g->reduceMean($g->mul($selected, $weights))
                );
                $probs = $g->softmax($logits);
                $entropy = $g->scale(-1.0, $g->reduceMean(
                    $g->reduceSum($g->mul($probs, $logProbs), axis:1)
                ));
                $totalLoss = $g->sub(
                    $policyLoss, $g->scale($this->entropyWeight, $entropy)
                );
                return [$totalLoss, $policyLoss, $entropy];
            }
        );
        $variables = $network->trainableVariables();
        $gradients = $this->clipGradients($tape->gradient($totalLoss, $variables));
        $this->optimizer->update($variables, $gradients);
        $network->syncWeightCaches();
        return ['policy_loss'=>$this->scalar($policyLoss), 'entropy'=>$this->scalar($entropy)];
    }

    private function scalar(object $value) : float
    {
        return (float)$this->la->scalar($value->value());
    }

    private function clipGradients(array $gradients) : array
    {
        return GradientClipping::clipByGlobalNorm(
            $this->la,$gradients,$this->maxGradNorm
        );
    }

    public function saveWeightsToFile(string $filepath, ?bool $portable = true) : void
    {
        $directory = dirname($filepath);
        if (!is_dir($directory) && !mkdir($directory, 0777, true) && !is_dir($directory)) {
            throw new \RuntimeException("Could not create checkpoint directory: {$directory}");
        }
        $weights = [];
        $this->network->saveWeights($weights, $portable);
        $checkpoint = ['format'=>'rindow-rl-reinforce', 'version'=>self::CHECKPOINT_VERSION,
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
        if (!is_array($checkpoint) || ($checkpoint['format'] ?? null) !== 'rindow-rl-reinforce'
            || ($checkpoint['version'] ?? null) !== self::CHECKPOINT_VERSION
            || ($checkpoint['obsDim'] ?? null) !== $this->obsDim
            || ($checkpoint['numActions'] ?? null) !== $this->numActions) {
            throw new \UnexpectedValueException("Invalid or incompatible REINFORCE checkpoint: {$filepath}");
        }
        $this->network->loadWeights($checkpoint['weights']);
    }
}
