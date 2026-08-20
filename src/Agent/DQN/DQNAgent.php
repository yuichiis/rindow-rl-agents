<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Util\GradientClipping;
use Rindow\RL\Agents\Util\ActionMask;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;

class DQNAgent
{
    private const CHECKPOINT_VERSION = 1;
    private object $la;
    private object $backend;
    private object $g;
    private object $optimizer;
    private int $updates = 0;
    public QNetwork $qNetwork;
    public QNetwork $targetNetwork;

    /**
     * @param int|array<int,int> $obsDim
     * @param array<int,int> $hiddenLayers
     * @param array<int,object>|null $featureLayers
     */
    public function __construct(
        private Builder $nn,
        private int|array $obsDim,
        private int $numActions,
        array $hiddenLayers=[128, 128],
        float $learningRate=1.0e-3,
        private float $gamma=0.99,
        private int $batchSize=64,
        private int $targetUpdateInterval=500,
        private float $maxGradNorm=10.0,
        private ?string $stateField=null,
        private ?string $actionMaskField=null,
        // Optional CNN/RNN feature extractor. The layers are cloned for the
        // online and target networks; the final action-value Dense is appended.
        ?array $featureLayers=null,
        private bool $ddqn=false,
    ) {
        if ($featureLayers === []) $featureLayers = null;
        $observationShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        if ($observationShape === []
            || array_filter($observationShape,static fn(int $dim)=>$dim < 1)
            || $numActions < 2 || $batchSize < 1 || $targetUpdateInterval < 1) {
            throw new \InvalidArgumentException('Invalid DQN dimensions or update parameters.');
        }
        $this->backend = $nn->backend();
        $this->la = $this->backend->primaryLA();
        $this->g = $nn->gradient();
        $this->obsDim = is_int($obsDim) ? $obsDim : $observationShape;
        $this->qNetwork = new QNetwork($nn,$this->obsDim,$numActions,$hiddenLayers,$featureLayers);
        $this->targetNetwork = new QNetwork($nn,$this->obsDim,$numActions,$hiddenLayers,$featureLayers);
        $dummy = $this->g->Variable($this->la->zeros(
            $this->la->alloc(array_merge([1],$observationShape), dtype:NDArray::float32)
        ));
        $this->qNetwork->forward($dummy);
        $this->targetNetwork->forward($dummy);
        $this->syncTargetNetwork();
        $this->optimizer = $nn->optimizers->Adam(lr:$learningRate);
    }

    public function summary() : void { $this->qNetwork->summary(); }
    public function observationDimension() : int { return array_product($this->observationShape()); }
    /** @return array<int> */
    public function observationShape() : array
    {
        return is_int($this->obsDim) ? [$this->obsDim] : $this->obsDim;
    }
    public function actionDimension() : int { return $this->numActions; }
    public function usesActionMask() : bool { return $this->actionMaskField !== null; }

    /**
     * @param NDArray|array<string,mixed> $observation
     * @return array{NDArray,?NDArray} network state and optional action mask
     */
    public function parseObservation(NDArray|array $observation) : array
    {
        if ($observation instanceof NDArray) {
            if ($this->stateField !== null || $this->actionMaskField !== null) {
                throw new \InvalidArgumentException('A dictionary observation was expected.');
            }
            return [$this->asNetworkState($observation),null];
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
                $mask = $this->la->astype($mask,dtype:NDArray::bool);
            }
            if (!ActionMask::hasAny($this->la,$mask)) {
                throw new \InvalidArgumentException('Action mask must allow at least one action.');
            }
        }
        return [$this->asNetworkState($state),$mask];
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
            ? $this->la->astype($state,dtype:NDArray::float32)
            : $state;
    }

    /** @param NDArray|array<string,mixed> $observation */
    public function selectAction(NDArray|array $observation, float $epsilon=0.0) : int
    {
        [$state,$mask] = $this->parseObservation($observation);
        return $this->selectActionFromState($state,$epsilon,$mask);
    }

    public function selectActionFromState(
        NDArray $observation,
        float $epsilon=0.0,
        ?NDArray $mask=null,
    ) : int
    {
        if ($epsilon < 0.0 || $epsilon > 1.0) {
            throw new \InvalidArgumentException('epsilon must be between zero and one.');
        }
        $randomArray = $this->la->randomUniform([1], 0.0, 1.0);
        $random = (float)$this->hostArray($randomArray)->toArray()[0];
        if ($random < $epsilon) {
            $allowed = $mask === null
                ? range(0,$this->numActions-1)
                : array_keys(array_filter(
                    $this->hostArray($mask)->toArray(),static fn($value)=>(bool)$value
                ));
            $indexArray = $this->la->randomUniform(
                [1],0,count($allowed)-1,dtype:NDArray::int32
            );
            $index = (int)$this->hostArray($indexArray)->toArray()[0];
            return $allowed[$index];
        }
        return $this->selectActionDeterministicFromState($observation,$mask);
    }

    /** @param NDArray|array<string,mixed> $observation */
    public function selectActionDeterministic(NDArray|array $observation) : int
    {
        [$state,$mask] = $this->parseObservation($observation);
        return $this->selectActionDeterministicFromState($state,$mask);
    }

    public function selectActionDeterministicFromState(
        NDArray $observation,
        ?NDArray $mask=null,
    ) : int
    {
        $batch = $this->la->copy($observation)->reshape(array_merge([1],$this->observationShape()));
        if ($this->la->isInt($batch)) $batch = $this->la->astype($batch, dtype:NDArray::float32);
        $qValues = $this->qNetwork->forward($this->g->Variable($batch),false)->value();
        if ($mask !== null) {
            $qValues = $this->la->masking(
                $this->la->expandDims($mask,axis:0),
                $this->la->copy($qValues),
                fill:-1.0e9,
            );
        }
        $best = $this->la->reduceArgMax(
            $qValues,axis:1,dtype:NDArray::int32
        );
        return (int)$this->la->scalar($best)[0];
    }

    /** @return array{loss:float,q_value:float} */
    public function update(ReplayBuffer $buffer) : array
    {
        [$observations,$actions,$rewards,$nextObservations,$dones,$nextActionMasks] =
            $buffer->sample($this->batchSize);
        $nextValues = $this->nextStateValues($nextObservations,$nextActionMasks);
        $notDones = $this->la->fill(
            1.0,
            $this->la->alloc($dones->shape(), dtype:$dones->dtype())
        );
        $notDones = $this->la->axpy($dones,$notDones,-1.0);
        $targets = $this->la->add(
            $rewards,
            $this->la->scal($this->gamma,$this->la->multiply($notDones,$nextValues))
        );
        $g = $this->g;
        $network = $this->qNetwork;
        [$loss,$meanQ] = $this->nn->with(
            $tape=$g->GradientTape(),
            function() use($g,$network,$observations,$actions,$targets) {
                $allQ = $network->forward($g->Variable($observations),true);
                $selectedQ = $g->gather($allQ,$actions,axis:1,batchDims:1);
                $loss = $g->reduceMean($g->square($g->sub($selectedQ,$g->constant($targets))));
                return [$loss,$g->reduceMean($selectedQ)];
            }
        );
        $variables = $network->trainableVariables();
        $gradients = $this->clipGradients($tape->gradient($loss,$variables));
        $this->optimizer->update($variables,$gradients);
        $this->updates++;
        if ($this->updates % $this->targetUpdateInterval === 0) $this->syncTargetNetwork();
        return ['loss'=>$this->scalar($loss),'q_value'=>$this->scalar($meanQ)];
    }

    private function nextStateValues(
        NDArray $nextObservations,
        ?NDArray $nextActionMasks,
    ) : NDArray {
        if ($this->ddqn) {
            // Double DQN selects the action with the online network and
            // evaluates that action with the target network.
            $onlineQ = $this->qNetwork->forward(
                $this->g->Variable($nextObservations),false
            )->value();
            if ($nextActionMasks !== null) {
                $onlineQ = $this->la->masking(
                    $nextActionMasks,$onlineQ,fill:-1.0e9
                );
            }
            $nextActions = $this->la->reduceArgMax(
                $onlineQ,axis:1,dtype:NDArray::int32
            );
            $targetQ = $this->targetNetwork->forward(
                $this->g->Variable($nextObservations),false
            )->value();
            return $this->la->gather($targetQ,$nextActions,axis:1);
        }

        // Standard DQN selects and evaluates with the target network.
        $targetQ = $this->targetNetwork->forward(
            $this->g->Variable($nextObservations),false
        )->value();
        if ($nextActionMasks !== null) {
            $targetQ = $this->la->masking(
                $nextActionMasks,$targetQ,fill:-1.0e9
            );
        }
        return $this->la->reduceMax($targetQ,axis:1);
    }

    private function syncTargetNetwork() : void
    {
        foreach ($this->qNetwork->trainableVariables() as $i=>$source) {
            $this->la->copy(
                $source->value(),
                $this->targetNetwork->trainableVariables()[$i]->value(),
            );
        }
    }

    /**
     * @param array<int,NDArray> $gradients
     * @return array<int,NDArray>
     */
    private function clipGradients(array $gradients) : array
    {
        return GradientClipping::clipByGlobalNorm(
            $this->la,$gradients,$this->maxGradNorm
        );
    }

    private function scalar(Variable $value) : float
    {
        return (float)$this->la->scalar($value->value());
    }

    private function hostArray(NDArray $value) : NDArray
    {
        return $this->backend->ndarray($value);
    }

    public function saveWeightsToFile(string $filepath, ?bool $portable=true) : void
    {
        $directory = dirname($filepath);
        if (!is_dir($directory) && !mkdir($directory,0777,true) && !is_dir($directory))
            throw new \RuntimeException("Could not create checkpoint directory: {$directory}");
        $weights = ['qNetwork'=>[],'targetNetwork'=>[]];
        $this->qNetwork->saveWeights($weights['qNetwork'],$portable);
        $this->targetNetwork->saveWeights($weights['targetNetwork'],$portable);
        $checkpoint = ['format'=>'rindow-rl-dqn','version'=>self::CHECKPOINT_VERSION,
            'obsDim'=>$this->obsDim,'numActions'=>$this->numActions,'weights'=>$weights];
        if (file_put_contents($filepath,serialize($checkpoint),LOCK_EX) === false)
            throw new \RuntimeException("Could not write checkpoint: {$filepath}");
    }

    public function loadWeightsFromFile(string $filepath) : void
    {
        if (!is_file($filepath)) throw new \InvalidArgumentException("Checkpoint does not exist: {$filepath}");
        $checkpoint = unserialize(file_get_contents($filepath),['allowed_classes'=>false]);
        if (!is_array($checkpoint) || ($checkpoint['format']??null)!=='rindow-rl-dqn'
            || ($checkpoint['version']??null)!==self::CHECKPOINT_VERSION
            || ($checkpoint['obsDim']??null)!==$this->obsDim
            || ($checkpoint['numActions']??null)!==$this->numActions)
            throw new \UnexpectedValueException("Invalid or incompatible DQN checkpoint: {$filepath}");
        $this->qNetwork->loadWeights($checkpoint['weights']['qNetwork']);
        $this->targetNetwork->loadWeights($checkpoint['weights']['targetNetwork']);
    }
}
