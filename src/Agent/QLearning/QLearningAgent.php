<?php
namespace Rindow\RL\Agents\Agent\QLearning;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;

/** Linear one-step Q-Learning with tile-coded observations. */
class QLearningAgent
{
    private const CHECKPOINT_VERSION = 1;
    /** @var array<int,array<int,float>> */
    private array $weights;
    private float $alpha;

    public function __construct(
        private object $la,
        private TileCoder $tileCoder,
        private int $numActions,
        float $learningRate = 0.3,
        private float $gamma = 0.99,
        private float $epsilon = 0.05,
        private ?string $stateField = null,
        private ?string $actionMaskField = null,
        float $initialValue = 0.0,
    ) {
        if ($numActions < 2 || $learningRate <= 0.0 || $gamma < 0.0 || $gamma > 1.0
            || $epsilon < 0.0 || $epsilon > 1.0) {
            throw new \InvalidArgumentException('Invalid Q-Learning parameters.');
        }
        if ($actionMaskField !== null && $stateField === null) {
            throw new \InvalidArgumentException('stateField is required when actionMaskField is used.');
        }
        $this->alpha = $learningRate / $tileCoder->activeFeatureCount();
        $initialWeight = $initialValue / $tileCoder->activeFeatureCount();
        $this->weights = array_fill(0, $numActions,
            array_fill(0, $tileCoder->featureCount(), $initialWeight));
    }

    public function observationDimension() : int { return $this->tileCoder->observationDimension(); }
    public function actionDimension() : int { return $this->numActions; }
    public function epsilon() : float { return $this->epsilon; }
    public function usesActionMask() : bool { return $this->actionMaskField !== null; }

    /** @return array{NDArray|array,?array<int,bool>} */
    public function parseObservation(NDArray|array $observation) : array
    {
        if ($observation instanceof NDArray) {
            if ($this->stateField !== null || $this->actionMaskField !== null) {
                throw new \InvalidArgumentException('A dictionary observation was expected.');
            }
            return [$observation, null];
        }
        if ($this->stateField === null) return [$observation, null];
        $state = $observation[$this->stateField] ?? null;
        if (!$state instanceof NDArray && !is_array($state)) {
            throw new \InvalidArgumentException(
                "Observation field '{$this->stateField}' must be an NDArray or array."
            );
        }
        $mask = null;
        if ($this->actionMaskField !== null) {
            $value = $observation[$this->actionMaskField] ?? null;
            if (!$value instanceof NDArray && !is_array($value)) {
                throw new \InvalidArgumentException(
                    "Observation field '{$this->actionMaskField}' must be an NDArray or array."
                );
            }
            $values = $value instanceof NDArray ? $value->toArray() : $value;
            if (count($values) !== $this->numActions) {
                throw new \InvalidArgumentException('Action mask size must equal numActions.');
            }
            $mask = array_map(static fn($item) : bool => (bool)$item, $values);
            if (!in_array(true, $mask, true)) {
                throw new \InvalidArgumentException('Action mask must allow at least one action.');
            }
        }
        return [$state, $mask];
    }

    public function value(NDArray|array $observation, int $action) : float
    {
        $this->validateAction($action);
        [$state] = $this->parseObservation($observation);
        return $this->valueOfFeatures($this->tileCoder->encode($state), $action);
    }

    public function selectAction(NDArray|array $observation, ?float $epsilon = null) : int
    {
        $epsilon ??= $this->epsilon;
        if ($epsilon < 0.0 || $epsilon > 1.0) {
            throw new \InvalidArgumentException('epsilon must be between zero and one.');
        }
        [$state, $mask] = $this->parseObservation($observation);
        $allowed = $this->allowedActions($mask);
        $random = (float)$this->la->randomUniform([1], 0.0, 1.0)->toArray()[0];
        if ($random < $epsilon) return $this->randomAction($allowed);
        return $this->greedyActionFromState($state, $allowed, true);
    }

    public function selectActionDeterministic(NDArray|array $observation) : int
    {
        [$state, $mask] = $this->parseObservation($observation);
        return $this->greedyActionFromState($state, $this->allowedActions($mask), false);
    }

    /** Performs one off-policy Q-Learning update and returns its TD error. */
    public function update(
        NDArray|array $observation,
        int $action,
        float $reward,
        NDArray|array $nextObservation,
        bool $terminal,
    ) : float {
        $this->validateAction($action);
        [$state, $mask] = $this->parseObservation($observation);
        if ($mask !== null && !$mask[$action]) {
            throw new \InvalidArgumentException('The selected action is disabled by the action mask.');
        }
        $features = $this->tileCoder->encode($state);
        $q = $this->valueOfFeatures($features, $action);
        $nextValue = 0.0;
        if (!$terminal) {
            [$nextState, $nextMask] = $this->parseObservation($nextObservation);
            $nextFeatures = $this->tileCoder->encode($nextState);
            $allowed = $this->allowedActions($nextMask);
            $nextValue = $this->valueOfFeatures($nextFeatures, $allowed[0]);
            foreach (array_slice($allowed, 1) as $nextAction) {
                $nextValue = max($nextValue,
                    $this->valueOfFeatures($nextFeatures, $nextAction));
            }
        }
        $delta = $reward + $this->gamma * $nextValue - $q;
        foreach ($features as $feature) {
            $this->weights[$action][$feature] += $this->alpha * $delta;
        }
        return $delta;
    }

    public function saveWeightsToFile(string $filepath) : void
    {
        $directory = dirname($filepath);
        if (!is_dir($directory) && !mkdir($directory, 0777, true) && !is_dir($directory)) {
            throw new \RuntimeException("Could not create checkpoint directory: {$directory}");
        }
        $checkpoint = ['format'=>'rindow-rl-tile-q-learning',
            'version'=>self::CHECKPOINT_VERSION, 'numActions'=>$this->numActions,
            'featureCount'=>$this->tileCoder->featureCount(), 'weights'=>$this->weights];
        if (file_put_contents($filepath, serialize($checkpoint), LOCK_EX) === false) {
            throw new \RuntimeException("Could not write checkpoint: {$filepath}");
        }
    }

    public function loadWeightsFromFile(string $filepath) : void
    {
        if (!is_file($filepath)) throw new \InvalidArgumentException("Checkpoint does not exist: {$filepath}");
        $checkpoint = unserialize(file_get_contents($filepath), ['allowed_classes'=>false]);
        if (!is_array($checkpoint) || ($checkpoint['format'] ?? null) !== 'rindow-rl-tile-q-learning'
            || ($checkpoint['version'] ?? null) !== self::CHECKPOINT_VERSION
            || ($checkpoint['numActions'] ?? null) !== $this->numActions
            || ($checkpoint['featureCount'] ?? null) !== $this->tileCoder->featureCount()) {
            throw new \UnexpectedValueException("Invalid or incompatible checkpoint: {$filepath}");
        }
        $this->weights = $checkpoint['weights'];
    }

    /** @param int[] $allowed */
    private function greedyActionFromState(NDArray|array $state, array $allowed, bool $randomTie) : int
    {
        $features = $this->tileCoder->encode($state);
        $best = [$allowed[0]];
        $bestValue = $this->valueOfFeatures($features, $allowed[0]);
        foreach (array_slice($allowed, 1) as $action) {
            $value = $this->valueOfFeatures($features, $action);
            if ($value > $bestValue) { $bestValue = $value; $best = [$action]; }
            elseif ($value === $bestValue) $best[] = $action;
        }
        return $randomTie ? $this->randomAction($best) : $best[0];
    }

    /** @param int[] $allowed */
    private function randomAction(array $allowed) : int
    {
        $index = (int)$this->la->randomUniform(
            [1], 0, count($allowed)-1, dtype:NDArray::int32
        )->toArray()[0];
        return $allowed[$index];
    }

    private function valueOfFeatures(array $features, int $action) : float
    {
        $value = 0.0;
        foreach ($features as $feature) $value += $this->weights[$action][$feature];
        return $value;
    }

    private function validateAction(int $action) : void
    {
        if ($action < 0 || $action >= $this->numActions) {
            throw new \InvalidArgumentException('Action is outside the discrete action space.');
        }
    }

    /** @param ?array<int,bool> $mask @return int[] */
    private function allowedActions(?array $mask) : array
    {
        if ($mask === null) return range(0, $this->numActions-1);
        return array_keys(array_filter($mask, static fn(bool $value) : bool => $value));
    }
}
