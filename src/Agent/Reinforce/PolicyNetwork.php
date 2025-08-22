<?php
namespace Rindow\RL\Agents\Agent\Reinforce;

use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Estimator\AbstractEstimatorNetwork;
use Rindow\RL\Agents\Util\Random;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Builder\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class PolicyNetwork extends AbstractEstimatorNetwork
{
    protected object $la;
    protected int $numActions;
    protected Model $policyModel;
    protected ?object $mo;  // for debug

    public function __construct(
        object $la,
        Builder $builder,
        array $stateShape,
        int $numActions,
        ?array $convLayers=null,
        ?string $convType=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ?string $lastActivation=null,
        ?string $lastKernelInitializer=null,
        ?Model $model=null,
        ?object $mo=null
        )
    {
        parent::__construct($builder,$stateShape);
        $this->la = $la;
        $this->numActions = $numActions;
        if($fcLayers===null) {
            $fcLayers = [32, 16];
        }
        if($model===null) {
            $model = $this->buildModel($stateShape,$numActions,
                $convLayers,$convType,$fcLayers,
                $activation,$kernelInitializer,$lastActivation,$lastKernelInitializer);
        }
        $this->policyModel = $model;
        $this->mo = $mo;
    }

    public function model() : Model
    {
        return $this->policyModel;
    }

    protected function buildModel(
        array $stateShape,
        int $numActions,
        ?array $convLayers=null,
        ?string $convType=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ?string $lastActivation=null,
        ?string $lastKernelInitializer=null,
    ) : Model
    {
        $nn = $this->builder;

        $model = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
        );

        $model->add($nn->layers->Dense(
            $numActions,
            activation:$lastActivation,
            kernel_initializer:$lastKernelInitializer,
        ));
        return $model;
    }

    protected function call(NDArray $inputs, mixed $training) : NDArray
    {
        $outputs = $this->policyModel->forward($inputs,$training);
        return $outputs;
    }

    /**
    * $states : (batches,...stateShape) typeof int32 or float32
    * $actionValues : (batches,...numActions)  typeof float32
    */
    public function getActionValues(NDArray $states,?bool $std=null) : NDArray|array
    {
        $la = $this->la;
        $orgStates = $states;
        if($la->isInt($states)) {
            $states = $la->astype($states,NDArray::float32);
        }
        $values = $this->predict($states);
        return $values;
    }

    public function __clone()
    {
        parent::__clone();
    }
}
