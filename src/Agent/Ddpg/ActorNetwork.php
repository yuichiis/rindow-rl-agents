<?php
namespace Rindow\RL\Agents\Agent\Ddpg;

use Rindow\RL\Agents\Estimator\AbstractEstimatorNetwork;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Builder\Builder;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class ActorNetwork extends AbstractEstimatorNetwork
{
    protected object $model;
    protected int $numActions;
    //protected $onesProb;

    public function __construct(
        Builder $builder,
        array $stateShape,
        int $numActions,
        ?array $convLayers=null,
        ?string $convType=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ?string $outputActivation=null,
        ?string $outputKernelInitializer=null,
        ?Model $model=null
        )
    {
        parent::__construct($builder,$stateShape);
        $this->numActions = $numActions;

        $la = $this->la;

        if($model===null) {
            $model = $this->buildActorModel(
                $stateShape,$numActions,
                $convLayers,$convType,$fcLayers,
                $activation,$kernelInitializer,
                $outputActivation,
                $outputKernelInitializer,
            );
        }
        $this->model = $model;
    }

    protected function buildActorModel(
        array $stateShape,
        int $numActions,
        ?array $convLayers=null,
        ?string $convType=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ?string $outputActivation=null,
        ?string $outputKernelInitializer=null,
        )
    {
        $nn = $this->builder;
        $K = $this->backend;

        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [32, 16];
        }

        $model = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );

        $outputActivation ??= 'tanh';
        $model->add($nn->layers->Dense(
            $numActions,
            activation:$outputActivation,
            kernel_initializer:$outputKernelInitializer,
        ));

        return $model;
    }

    protected function call(NDArray $inputs, ?bool $training) : NDArray
    {
        $outputs = $this->model->forward($inputs,$training);
        return $outputs;
    }

    public function __clone()
    {
        parent::__clone();
    }
}