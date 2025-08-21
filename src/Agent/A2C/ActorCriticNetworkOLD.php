<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Estimator\AbstractNetwork;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Builder\Builder;

class ActorCriticNetwork extends AbstractNetwork implements Estimator
{
    protected int $numActions;
    protected Model $stateLayers;
    protected Layer $actionLayer;
    protected Layer $criticLayer;

    public function __construct(
        object $la,
        Builder $builder,
        array $stateShape, int $numActions,
        ?array $convLayers=null,?string $convType=null,?array $fcLayers=null,
        ?string $activation=null, ?string $kernelInitializer=null,
        ?string $actionActivation=null, ?string $actionKernelInitializer=null,
        ?string $criticKernelInitializer=null,
        )
    {
        parent::__construct($builder,$stateShape);
        $this->la = $la;
        $nn = $this->builder();
        $this->numActions = $numActions;

        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [64, 64];
        }
        if($activation===null) {
            $activation = 'tanh'; #'relu';
        }
        //if($actionActivation===null) {
        //    $actionActivation = 'softmax';
        //}
        $this->stateLayers = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
            name:'State'
        );
        $this->actionLayer = $nn->layers()->Dense(
            $numActions,
            activation:$actionActivation,
            kernel_initializer:$actionKernelInitializer,
            name:'Action'
        );
        $this->criticLayer = $nn->layers()->Dense(
            1,
            kernel_initializer:$criticKernelInitializer,
            name:'Critic'
        );
        $this->numActions = $numActions;
    }

    public function call(NDArray $state_input, mixed $training=null) : array
    {
        $state_out = $this->stateLayers->forward($state_input,$training);
        $action_out = $this->actionLayer->forward($state_out,$training);
        $critic_out = $this->criticLayer->forward($state_out,$training);
        return [$action_out,$critic_out];
    }

    /**
     * @param  NDArray $states : (batches,...StateDims) typeof int32 or float32
     * @return NDArray $actionValues : (batches,...ValueDims) typeof float32
     */
    public function getActionValues(NDArray $states) : NDArray
    {
        $la = $this->la;
        if($states->ndim()<2) {
            $specs = $la->dtypeToString($states->dtype())."(".implode(',',$states->shape()).")";
            throw new InvalidArgumentException("states must be a 2-dimensional array or higher. $specs given.");
        }
        $orgStates = $states;
        if($la->isInt($states)) {
            $states = $la->astype($states,NDArray::float32);
        }

        [$action_out,$critic_out] = $this->forward($states,false);
        return $action_out;
    }
}
