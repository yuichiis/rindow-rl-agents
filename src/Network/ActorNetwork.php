<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;

class ActorNetwork extends AbstractNetwork implements QPolicy
{
    protected $actionSize;

    public function __construct($la,$builder,
            array $obsSize, array $actionSize,
            array $convLayers=null,string $convType=null,array $fcLayers=null,
            $activation=null,$kernelInitializer=null,
            array $outputOptions=null,
            $model=null
        )
    {
        parent::__construct($builder,$obsSize);
        $this->la = $la;
        $this->actionSize = $actionSize;

        if($model===null) {
            $model = $this->buildActorModel(
                $obsSize,$actionSize,
                $convLayers,$convType,$fcLayers,
                $activation,$kernelInitializer,
                $outputOptions,
            );
        }
        $this->model = $model;
    }

    public function actionSize()
    {
        return $this->actionSize;
    }

    protected function buildActorModel(
        $obsSize,$actionSize,
        $convLayers,$convType,$fcLayers,
        $activation,$kernelInitializer,
        $outputOptions,
        )
    {
        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [256, 256];
        }

        $nn = $this->builder;
        $K = $this->backend;
        $model = $this->buildMlpLayers(
            $obsSize,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer);

        $actionSize = (int)array_product($actionSize);
        $last_init = null;
        $initializer = null;
        $initializerOpts = [];
        if(isset($outputOptions['initializer']['kernelInitializer'])) {
            $initializer = $outputOptions['initializer']['kernelInitializer'];
        }
        if(isset($outputOptions['initializer']['options'])) {
            $initializerOpts = $outputOptions['initializer']['options'];
        }
        if($initializer !== null) {
            $last_init = $K->getInitializer($initializer, ...$initializerOpts);
        }
        $last_activation = 'softmax';
        if(isset($outputOptions['activation'])) {
            $last_activation = $outputOptions['activation'];
        }
        $model->add(
            $nn->layers->Dense($actionSize, activation:$last_activation, kernel_initializer:$last_init)
        );
        return $model;
    }

    protected function call($inputs,$training)
    {
        $outputs = $this->model->forward($inputs,$training);
        return $outputs;
    }

    public function getQValues($state) : NDArray
    {
        $la = $this->la;
        if(is_numeric($state)) {
            $state = $this->la->array([$state]);
        } elseif(!($state instanceof NDArray)) {
            throw new InvalidArgumentException('Observation must be NDArray');
        }
        $state = $la->expandDims($state,$axis=0);
        //$actions = $actor_model->predict($state);
        $actions = $this->model->forward($state,false);
        $actions = $la->squeeze($actions,$axis=0);
        return $actions;
    }

    public function sample($state)
    {
        throw new LogicException('Unsupported operation');
    }
}