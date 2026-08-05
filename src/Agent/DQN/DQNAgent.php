<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;

class DQNAgent
{
    private const CHECKPOINT_VERSION = 1;
    private object $la;
    private object $g;
    private object $optimizer;
    private int $updates = 0;
    public QNetwork $qNetwork;
    public QNetwork $targetNetwork;

    public function __construct(
        private Builder $nn,
        private int $obsDim,
        private int $numActions,
        array $hiddenLayers=[128, 128],
        float $learningRate=1.0e-3,
        private float $gamma=0.99,
        private int $batchSize=64,
        private int $targetUpdateInterval=500,
        private float $maxGradNorm=10.0,
    ) {
        if ($obsDim < 1 || $numActions < 2 || $batchSize < 1 || $targetUpdateInterval < 1) {
            throw new \InvalidArgumentException('Invalid DQN dimensions or update parameters.');
        }
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        $this->qNetwork = new QNetwork($nn,$obsDim,$numActions,$hiddenLayers);
        $this->targetNetwork = new QNetwork($nn,$obsDim,$numActions,$hiddenLayers);
        $dummy = $this->g->Variable($this->la->zeros(
            $this->la->alloc([1,$obsDim], dtype:NDArray::float32)
        ));
        $this->qNetwork->forward($dummy);
        $this->targetNetwork->forward($dummy);
        $this->syncTargetNetwork();
        $this->optimizer = $nn->optimizers->Adam(lr:$learningRate);
    }

    public function summary() : void { $this->qNetwork->summary(); }
    public function observationDimension() : int { return $this->obsDim; }
    public function actionDimension() : int { return $this->numActions; }

    public function selectAction(NDArray $observation, float $epsilon=0.0) : int
    {
        if ($epsilon < 0.0 || $epsilon > 1.0) {
            throw new \InvalidArgumentException('epsilon must be between zero and one.');
        }
        $random = (float)$this->la->randomUniform([1], 0.0, 1.0)->toArray()[0];
        if ($random < $epsilon) {
            return (int)$this->la->randomUniform([1], 0, $this->numActions-1,
                dtype:NDArray::int32)->toArray()[0];
        }
        return $this->selectActionDeterministic($observation);
    }

    public function selectActionDeterministic(NDArray $observation) : int
    {
        $batch = $this->la->copy($observation)->reshape([1,$this->obsDim]);
        if ($this->la->isInt($batch)) $batch = $this->la->astype($batch, dtype:NDArray::float32);
        $values = $this->qNetwork->forward($this->g->Variable($batch),false)->value()[0]->toArray();
        $best = 0;
        foreach ($values as $action=>$value) {
            if ($value > $values[$best]) $best = $action;
        }
        return $best;
    }

    /** @return array{loss:float,q_value:float} */
    public function update(ReplayBuffer $buffer) : array
    {
        [$observations,$actions,$rewards,$nextObservations,$dones] =
            $buffer->sample($this->batchSize);
        $nextQ = $this->targetNetwork->forward($this->g->Variable($nextObservations),false)->value();
        $nextValues = $this->la->reduceMax($nextQ, axis:1);
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
        $network->syncWeightCaches();
        $this->updates++;
        if ($this->updates % $this->targetUpdateInterval === 0) $this->syncTargetNetwork();
        return ['loss'=>$this->scalar($loss),'q_value'=>$this->scalar($meanQ)];
    }

    private function syncTargetNetwork() : void
    {
        foreach ($this->qNetwork->trainableVariables() as $i=>$source) {
            $this->targetNetwork->trainableVariables()[$i]->assign($source);
        }
        $this->targetNetwork->syncWeightCaches();
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
                else $sumSquares += (float)$value*(float)$value;
            }
        }
        $norm = sqrt($sumSquares);
        if ($norm <= $this->maxGradNorm || $norm == 0.0) return $gradients;
        $scale = $this->maxGradNorm/($norm+1.0e-8);
        foreach ($gradients as $i=>$gradient) $gradients[$i] = $this->la->scal($scale,$gradient);
        return $gradients;
    }

    private function scalar(Variable $value) : float
    {
        $array = $value->value()->toArray();
        while (is_array($array)) $array = reset($array);
        return (float)$array;
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
