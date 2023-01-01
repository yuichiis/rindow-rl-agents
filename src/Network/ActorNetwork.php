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
    protected $onesProb;

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
        $p = $la->alloc([1,(int)array_product($actionSize)]);
        $this->onesProb = $la->ones($p);
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

    public function getQValues(NDArray $states) : NDArray
    {
        $la = $this->la;
        $values = $this->model->forward($states,false);
        if($this->masks) {
            $numClass = count($actions);
            $states = $la->squeeze($states,$axis=-1);
            $masks = $la->gather($this->masks,$states,$axis=null);
            $la->multiply($masks,$values);
            $la->nan2num($values,-INF);
        }
        return $values;
    }

    public function samples(NDArray $states) : NDArray
    {
        $la = $this->la;
        if($this->masks) {
            $obs = $la->squeeze($states,$axis=-1);
            $values = $la->gather($this->probabilities,$obs,$axis=null);
        } else {
            $values = $la->repeat($this->onesProb,count($states),$axis=null);
        }
        return $values;
    }
}