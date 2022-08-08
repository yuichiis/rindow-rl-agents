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
            float $minval=null, float $maxval=null,
            $model=null
        )
    {
        parent::__construct($builder,$obsSize);
        $this->la = $la;
        $this->actionSize = $actionSize;

        if($model===null) {
            $model = $this->buildQModel(
                $obsSize,$actionSize,
                $convLayers,$convType,$fcLayers,
                $activation,$kernelInitializer,
                $minval, $maxval
            );
        }
        $this->qmodel = $model;
    }

    public function actionSize()
    {
        return $this->actionSize;
    }

    protected function buildQModel(
        $obsSize,$actionSize,
        $convLayers,$convType,$fcLayers,
        $activation,$kernelInitializer,
        $minval, $maxval
        )
    {
        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [256, 256];
        }
        $minval = $minval ?? -0.003;
        $maxval = $maxval ?? 0.003;

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
        $last_init = $K->getInitializer('random_uniform',
                minval:$minval,maxval:$maxval);
        $model->add(
            $nn->layers->Dense(1, activation:'tanh', kernel_initializer:$last_init)
        );
        return $model;
    }

    protected function call($inputs,$training)
    {
        $outputs = $this->qmodel->forward($inputs,$training);
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
        $actions = $this->qmodel->forward($state,false);
        $actions = $la->squeeze($actions,$axis=0);
        return $actions;
    }

    public function sample($state)
    {
        throw new LogicException('Unsupported operation');
    }
}