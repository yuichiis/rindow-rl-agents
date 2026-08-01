<?php
namespace Rindow\RL\Agents\Agent\PPO;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;

/** CartPoleなどの離散行動環境で使うActor-Criticネットワーク。 */
class ActorCritic extends AbstractModel
{
    protected ?AbstractModel $trunk = null;
    protected ?AbstractModel $actor = null;
    protected ?AbstractModel $critic = null;
    protected ?object $actorHead = null;
    protected ?object $criticHead = null;

    public function __construct(
        Builder $nn,
        int $obsDim,
        int $numActions,
        array $hiddenLayers = [64, 64],
        private bool $sharedBackbone = false,
    ) {
        parent::__construct($nn);
        if (!$sharedBackbone) {
            $this->actor = $this->mlp($nn, $obsDim, $hiddenLayers, $numActions, 'tanh');
            $this->critic = $this->mlp($nn, $obsDim, $hiddenLayers, 1, 'tanh');
            return;
        }
        $layers = [];
        foreach ($hiddenLayers as $i => $units) {
            // The legacy ActorCriticNetwork's MLP uses he_uniform for its
            // ReLU hidden layers.  Keep this explicit; Dense's library
            // default is different and changes the seeded initial policy.
            $options = ['activation' => 'relu', 'kernel_initializer' => 'he_uniform'];
            if ($i === 0) {
                $options['input_shape'] = [$obsDim];
            }
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $this->trunk = $nn->models->Sequential($layers);
        $this->actorHead = $nn->layers->Dense($numActions);
        $this->criticHead = $nn->layers->Dense(1);
    }

    private function mlp(
        Builder $nn,
        int $obsDim,
        array $hiddenLayers,
        int $outputDim,
        string $activation,
    ) : AbstractModel {
        $layers = [];
        foreach ($hiddenLayers as $i => $units) {
            $options = ['activation' => $activation];
            if ($i === 0) {
                $options['input_shape'] = [$obsDim];
            }
            $layers[] = $nn->layers->Dense($units, ...$options);
        }
        $layers[] = $nn->layers->Dense($outputDim);
        return $nn->models->Sequential($layers);
    }

    private function features(Variable $observations, ?bool $training = null) : Variable
    {
        return $this->trunk->forward($observations, $training);
    }

    public function policy(Variable $observations, ?bool $training = null) : Variable
    {
        if (!$this->sharedBackbone) {
            return $this->actor->forward($observations, $training);
        }
        return $this->actorHead->forward($this->features($observations, $training), $training);
    }

    public function value(Variable $observations, ?bool $training = null) : Variable
    {
        if (!$this->sharedBackbone) {
            return $this->critic->forward($observations, $training);
        }
        return $this->criticHead->forward($this->features($observations, $training), $training);
    }

    public function call(Variable $observations, ?bool $training = null) : array
    {
        if (!$this->sharedBackbone) {
            return [
                $this->actor->forward($observations, $training),
                $this->critic->forward($observations, $training),
            ];
        }
        $features = $this->features($observations, $training);
        return [
            $this->actorHead->forward($features, $training),
            $this->criticHead->forward($features, $training),
        ];
    }

    public function syncWeightCaches() : void
    {
        $models = $this->sharedBackbone
            ? [$this->trunk, $this->actorHead, $this->criticHead]
            : [$this->actor, $this->critic];
        foreach ($models as $model) {
            foreach ($model->submodules() as $module) {
                $module->reverseSyncWeightVariables();
            }
        }
    }
}
