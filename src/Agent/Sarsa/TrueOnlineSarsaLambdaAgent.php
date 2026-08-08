<?php
namespace Rindow\RL\Agents\Agent\Sarsa;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;

/** Linear True Online Sarsa(lambda) with tile-coded continuous observations. */
class TrueOnlineSarsaLambdaAgent
{
    private const CHECKPOINT_VERSION = 1;
    /** @var array<int,array<int,float>> */
    private array $weights;
    /** @var array<int,array<int,float>> */
    private array $traces;
    private float $qOld = 0.0;
    private float $alpha;

    public function __construct(
        private object $la,
        private TileCoder $tileCoder,
        private int $numActions,
        float $learningRate = 0.3,
        private float $gamma = 1.0,
        private float $lambda = 0.9,
        private float $epsilon = 0.0,
        private ?string $stateField = null,
        private ?string $actionMaskField = null,
        float $initialValue = 0.0,
        private ?Builder $nn = null,
    ) {
        if ($numActions < 2 || $learningRate <= 0.0 || $gamma < 0.0 || $gamma > 1.0
            || $lambda < 0.0 || $lambda > 1.0 || $epsilon < 0.0 || $epsilon > 1.0) {
            throw new \InvalidArgumentException('Invalid True Online Sarsa(lambda) parameters.');
        }
        if ($actionMaskField !== null && $stateField === null) {
            throw new \InvalidArgumentException('stateField is required when actionMaskField is used.');
        }
        // Tile coding activates numTilings features, so divide the nominal step size.
        $this->alpha = $learningRate / $tileCoder->activeFeatureCount();
        $initialWeight = $initialValue / $tileCoder->activeFeatureCount();
        $this->weights = array_fill(0, $numActions,
            array_fill(0, $tileCoder->featureCount(), $initialWeight));
        $this->traces = $this->emptyTable();
    }

    public function observationDimension() : int { return $this->tileCoder->observationDimension(); }
    public function actionDimension() : int { return $this->numActions; }
    public function epsilon() : float { return $this->epsilon; }
    public function usesActionMask() : bool { return $this->actionMaskField !== null; }

    /** @return array{NDArray|array,?array<int,bool>} tile-coder state and action mask */
    public function parseObservation(NDArray|array $observation) : array
    {
        if ($observation instanceof NDArray) {
            if ($this->stateField !== null || $this->actionMaskField !== null) {
                throw new \InvalidArgumentException('A dictionary observation was expected.');
            }
            return [$observation, null];
        }
        if ($this->stateField === null) {
            return [$observation, null];
        }
        $state = $observation[$this->stateField] ?? null;
        if (!$state instanceof NDArray && !is_array($state)) {
            throw new \InvalidArgumentException(
                "Observation field '{$this->stateField}' must be an NDArray or array."
            );
        }
        $mask = null;
        if ($this->actionMaskField !== null) {
            $maskValue = $observation[$this->actionMaskField] ?? null;
            if (!$maskValue instanceof NDArray && !is_array($maskValue)) {
                throw new \InvalidArgumentException(
                    "Observation field '{$this->actionMaskField}' must be an NDArray or array."
                );
            }
            $maskValues = $maskValue instanceof NDArray
                ? $this->hostArray($maskValue)->toArray() : $maskValue;
            if (count($maskValues) !== $this->numActions) {
                throw new \InvalidArgumentException('Action mask size must equal numActions.');
            }
            $mask = array_map(static fn($value) : bool => (bool)$value, $maskValues);
            if (!in_array(true, $mask, true)) {
                throw new \InvalidArgumentException('Action mask must allow at least one action.');
            }
        }
        return [$state, $mask];
    }

    public function startEpisode() : void
    {
        $this->traces = $this->emptyTable();
        $this->qOld = 0.0;
    }

    public function value(NDArray|array $observation, int $action) : float
    {
        $this->validateAction($action);
        [$state] = $this->parseObservation($observation);
        return $this->valueOfFeatures(
            $this->tileCoder->encode($this->hostArray($state)), $action
        );
    }

    public function selectAction(NDArray|array $observation, ?float $epsilon = null) : int
    {
        $epsilon ??= $this->epsilon;
        if ($epsilon < 0.0 || $epsilon > 1.0) {
            throw new \InvalidArgumentException('epsilon must be between zero and one.');
        }
        [$state, $mask] = $this->parseObservation($observation);
        $allowed = $this->allowedActions($mask);
        $random = $this->scalar($this->la->randomUniform([1], 0.0, 1.0));
        if ($random < $epsilon) {
            $index = (int)$this->scalar($this->la->randomUniform(
                [1], 0, count($allowed) - 1, dtype:NDArray::int32
            ));
            return $allowed[$index];
        }
        return $this->greedyActionFromState($state, $allowed, true);
    }

    public function selectActionDeterministic(NDArray|array $observation) : int
    {
        [$state, $mask] = $this->parseObservation($observation);
        return $this->greedyActionFromState($state, $this->allowedActions($mask), false);
    }

    /** @param int[] $allowed */
    private function greedyActionFromState(
        NDArray|array $state,
        array $allowed,
        bool $randomTie,
    ) : int
    {
        $features = $this->tileCoder->encode($this->hostArray($state));
        $bestActions = [$allowed[0]];
        $bestValue = $this->valueOfFeatures($features, $allowed[0]);
        foreach (array_slice($allowed, 1) as $action) {
            $value = $this->valueOfFeatures($features, $action);
            if ($value > $bestValue) {
                $bestValue = $value;
                $bestActions = [$action];
            } elseif ($value === $bestValue) {
                $bestActions[] = $action;
            }
        }
        if (!$randomTie) return $bestActions[0];
        // Random tie breaking prevents the initially zero training policy favouring action zero.
        $index = (int)$this->scalar($this->la->randomUniform(
            [1], 0, count($bestActions) - 1, dtype:NDArray::int32
        ));
        return $bestActions[$index];
    }

    /** Performs one transition update and returns its TD error. */
    public function update(
        NDArray|array $observation,
        int $action,
        float $reward,
        NDArray|array $nextObservation,
        ?int $nextAction,
        bool $terminal,
    ) : float {
        $this->validateAction($action);
        if (!$terminal && $nextAction === null) {
            throw new \InvalidArgumentException('nextAction is required for a non-terminal update.');
        }
        [$state, $mask] = $this->parseObservation($observation);
        if ($mask !== null && !$mask[$action]) {
            throw new \InvalidArgumentException('The selected action is disabled by the action mask.');
        }
        $features = $this->tileCoder->encode($this->hostArray($state));
        $q = $this->valueOfFeatures($features, $action);
        $qNext = $terminal ? 0.0 : $this->value($nextObservation, $nextAction);
        $delta = $reward + $this->gamma * $qNext - $q;

        $decay = $this->gamma * $this->lambda;
        $dot = 0.0;
        foreach ($features as $feature) $dot += $this->traces[$action][$feature] ?? 0.0;
        foreach ($this->traces as &$actionTraces) {
            foreach ($actionTraces as $feature => $trace) {
                $trace *= $decay;
                if (abs($trace) < 1.0e-12) unset($actionTraces[$feature]);
                else $actionTraces[$feature] = $trace;
            }
        }
        unset($actionTraces);
        $dutchIncrement = 1.0 - $this->alpha * $decay * $dot;
        foreach ($features as $feature) {
            $this->traces[$action][$feature] = ($this->traces[$action][$feature] ?? 0.0)
                + $dutchIncrement;
        }

        $correction = $q - $this->qOld;
        foreach ($this->weights as $weightAction => &$actionWeights) {
            foreach ($this->traces[$weightAction] as $feature => $trace) {
                $actionWeights[$feature] += $this->alpha
                    * ($delta + $correction) * $trace;
            }
        }
        unset($actionWeights);
        foreach ($features as $feature) {
            $this->weights[$action][$feature] -= $this->alpha * $correction;
        }
        $this->qOld = $terminal ? 0.0 : $qNext;
        return $delta;
    }

    public function saveWeightsToFile(string $filepath) : void
    {
        $directory = dirname($filepath);
        if (!is_dir($directory) && !mkdir($directory, 0777, true) && !is_dir($directory)) {
            throw new \RuntimeException("Could not create checkpoint directory: {$directory}");
        }
        $checkpoint = ['format'=>'rindow-rl-true-online-sarsa-lambda',
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
        if (!is_array($checkpoint) || ($checkpoint['format'] ?? null) !== 'rindow-rl-true-online-sarsa-lambda'
            || ($checkpoint['version'] ?? null) !== self::CHECKPOINT_VERSION
            || ($checkpoint['numActions'] ?? null) !== $this->numActions
            || ($checkpoint['featureCount'] ?? null) !== $this->tileCoder->featureCount()) {
            throw new \UnexpectedValueException("Invalid or incompatible checkpoint: {$filepath}");
        }
        $this->weights = $checkpoint['weights'];
        $this->startEpisode();
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
        if ($mask === null) return range(0, $this->numActions - 1);
        return array_keys(array_filter($mask, static fn(bool $allowed) : bool => $allowed));
    }

    private function emptyTable() : array
    {
        return array_fill(0, $this->numActions, []);
    }

    private function hostArray(NDArray|array $value) : NDArray|array
    {
        if (!$value instanceof NDArray || $this->nn === null) {
            return $value;
        }
        return $this->nn->hostArray($value);
    }

    private function scalar(NDArray $value) : float
    {
        $value = $this->la->scalar($value);
        while (is_array($value)) $value = reset($value);
        return (float)$value;
    }
}
