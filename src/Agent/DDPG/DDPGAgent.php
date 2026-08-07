<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

class DDPGAgent
{
    private const CHECKPOINT_VERSION = 1;
    private object $la;
    private object $g;
    private object $actorOptimizer;
    private object $criticOptimizer;
    public Actor $actor;
    public Actor $actorTarget;
    public Critic $critic;
    public Critic $criticTarget;

    public function __construct(
        private Builder $nn,
        private int|array $obsDim,
        private int $actDim,
        private float $actLimit,
        int $hiddenDim=256,
        float $lrActor=1.0e-4,
        float $lrCritic=1.0e-3,
        private float $gamma=0.99,
        private float $tau=0.005,
        private int $batchSize=128,
        // Optional CNN/RNN feature extractor. Layers are cloned independently
        // for actor, critic, and both target networks.
        ?array $featureLayers=null,
    ) {
        if ($featureLayers === []) $featureLayers = null;
        $observationShape = is_int($obsDim) ? [$obsDim] : array_values($obsDim);
        if ($observationShape === []
            || array_filter($observationShape,static fn($dim)=>!is_int($dim) || $dim < 1)
            || $actDim < 1 || $actLimit <= 0.0 || $batchSize < 1) {
            throw new \InvalidArgumentException('Invalid DDPG dimensions or update parameters.');
        }
        $this->obsDim = is_int($obsDim) ? $obsDim : $observationShape;
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        $this->actor = new Actor($nn,$this->obsDim,$actDim,$hiddenDim,$featureLayers);
        $this->actorTarget = new Actor($nn,$this->obsDim,$actDim,$hiddenDim,$featureLayers);
        $this->critic = new Critic($nn,$this->obsDim,$actDim,$hiddenDim,$featureLayers);
        $this->criticTarget = new Critic($nn,$this->obsDim,$actDim,$hiddenDim,$featureLayers);

        $obs = $this->g->Variable($this->la->zeros($this->la->alloc(
            array_merge([1],$observationShape),dtype:NDArray::float32
        )));
        $act = $this->g->Variable($this->la->zeros($this->la->alloc([1,$actDim], dtype:NDArray::float32)));
        $this->actor->forward($obs);
        $this->actorTarget->forward($obs);
        $this->critic->forward($obs,$act);
        $this->criticTarget->forward($obs,$act);
        $this->softUpdate($this->actor,$this->actorTarget,1.0);
        $this->softUpdate($this->critic,$this->criticTarget,1.0);
        $this->actorOptimizer = $nn->optimizers->Adam(lr:$lrActor);
        $this->criticOptimizer = $nn->optimizers->Adam(lr:$lrCritic);
    }

    public function summary() : void
    {
        echo "***** Actor Network *****\n";
        $this->actor->summary();
        echo "\n***** Critic Network *****\n";
        $this->critic->summary();
    }

    /** @return array<int> */
    public function observationShape() : array
    {
        return is_int($this->obsDim) ? [$this->obsDim] : $this->obsDim;
    }

    public function selectActionDeterministic(NDArray $obs) : NDArray
    {
        if ($obs->shape() !== $this->observationShape()) {
            throw new \InvalidArgumentException(sprintf(
                'Observation shape must be [%s]; [%s] given.',
                implode(',',$this->observationShape()),implode(',',$obs->shape())
            ));
        }
        $batch = $this->la->copy($obs)->reshape(array_merge([1],$this->observationShape()));
        if ($this->la->isInt($batch)) $batch = $this->la->astype($batch,dtype:NDArray::float32);
        $batch = $this->g->Variable($batch);
        $action = $this->actor->forward($batch)->value()->reshape([$this->actDim]);
        return $this->clip($this->la->scal($this->actLimit,$action));
    }

    public function selectAction(NDArray $obs, ?NDArray $noise=null) : NDArray
    {
        $action = $this->selectActionDeterministic($obs);
        if ($noise !== null) $action = $this->la->add($action,$noise);
        return $this->clip($action);
    }

    public function update(ReplayBuffer $buffer) : array
    {
        [$obs,$actions,$rewards,$nextObs,$dones] = $buffer->sample($this->batchSize);
        $g = $this->g;
        $obsV = $g->Variable($obs); $actionsV = $g->Variable($actions);
        $rewardsV = $g->Variable($rewards); $nextObsV = $g->Variable($nextObs);
        $donesV = $g->Variable($dones);

        $nextActions = $g->mul($this->actorTarget->forward($nextObsV),$this->actLimit);
        $nextQ = $this->criticTarget->forward($nextObsV,$nextActions);
        $targetQ = $g->stopGradient($g->add($rewardsV,
            $g->mul($this->gamma,$g->mul($g->sub(1.0,$donesV),$nextQ))));

        $critic = $this->critic;
        $criticLoss = $this->nn->with($criticTape=$g->GradientTape(), function() use($g,$critic,$obsV,$actionsV,$targetQ) {
            $q = $critic->forward($obsV,$actionsV,true);
            return $g->reduceMean($g->square($g->sub($q,$targetQ)));
        });
        $criticVars = $critic->trainableVariables();
        $this->criticOptimizer->update($criticVars,$criticTape->gradient($criticLoss,$criticVars));
        $critic->syncWeightCaches();

        $actor = $this->actor; $actLimit = $this->actLimit;
        $actorLoss = $this->nn->with($actorTape=$g->GradientTape(), function() use($g,$actor,$critic,$obsV,$actLimit) {
            $policyActions = $g->mul($actor->forward($obsV,true),$actLimit);
            return $g->scale(-1.0,$g->reduceMean($critic->forward($obsV,$policyActions,false)));
        });
        $actorVars = $actor->trainableVariables();
        $this->actorOptimizer->update($actorVars,$actorTape->gradient($actorLoss,$actorVars));
        $actor->syncWeightCaches();

        $this->softUpdate($this->actor,$this->actorTarget,$this->tau);
        $this->softUpdate($this->critic,$this->criticTarget,$this->tau);
        $this->actorTarget->syncWeightCaches();
        $this->criticTarget->syncWeightCaches();
        return ['actor_loss'=>$this->scalar($actorLoss), 'critic_loss'=>$this->scalar($criticLoss)];
    }

    private function softUpdate(AbstractModel $source, AbstractModel $target, float $tau) : void
    {
        foreach ($source->trainableVariables() as $i=>$sourceWeight) {
            $targetWeight = $target->trainableVariables()[$i];
            $targetWeight->assign($this->g->add($this->g->scale($tau,$sourceWeight),
                $this->g->scale(1.0-$tau,$targetWeight)));
        }
    }

    private function clip(NDArray $action) : NDArray
    {
        return $this->la->minimum($this->la->maximum($action,-$this->actLimit),$this->actLimit);
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
        $weights = [];
        foreach (['actor','actorTarget','critic','criticTarget'] as $name) {
            $weights[$name] = []; $this->{$name}->saveWeights($weights[$name],$portable);
        }
        $checkpoint = ['format'=>'rindow-rl-ddpg','version'=>self::CHECKPOINT_VERSION,
            'obsDim'=>$this->obsDim,'actDim'=>$this->actDim,'weights'=>$weights];
        if (file_put_contents($filepath,serialize($checkpoint),LOCK_EX) === false)
            throw new \RuntimeException("Could not write checkpoint: {$filepath}");
    }

    public function loadWeightsFromFile(string $filepath) : void
    {
        if (!is_file($filepath)) throw new \InvalidArgumentException("Checkpoint does not exist: {$filepath}");
        $checkpoint = unserialize(file_get_contents($filepath),['allowed_classes'=>false]);
        if (!is_array($checkpoint) || ($checkpoint['format']??null)!=='rindow-rl-ddpg'
            || ($checkpoint['version']??null)!==self::CHECKPOINT_VERSION
            || ($checkpoint['obsDim']??null)!==$this->obsDim || ($checkpoint['actDim']??null)!==$this->actDim)
            throw new \UnexpectedValueException("Invalid or incompatible DDPG checkpoint: {$filepath}");
        foreach (['actor','actorTarget','critic','criticTarget'] as $name) $this->{$name}->loadWeights($checkpoint['weights'][$name]);
    }
}
