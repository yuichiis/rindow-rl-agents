<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;

class ActorNetwork extends AbstractNetwork implements QPolicy
{
    use Random;

    protected $la;
    protected $model;
    protected $probabilities;
    protected $actionSize;
    protected $onesProb;
    protected $masks;

    public function __construct($la,$builder,
            array $obsSize, array $actionSize,
            array $convLayers=null,string $convType=null,array $fcLayers=null,
            $activation=null,$kernelInitializer=null,
            array $outputOptions=null,
            NDArray $rules=null,$model=null
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
        $this->masks = $rules;
        $this->model = $model;
        if($rules) {
            if($rules->shape()[1]!=array_product($actionSize)) {
                throw new InvalidArgumentException('The rules must match numActions');
            }
            $p = $this->generateProbabilities($rules);
            $this->probabilities = $p;
            //$this->thresholds = $this->generateThresholds($p);
            $this->masks = $rules;
        } else {
            $p = $la->alloc([1,(int)array_product($actionSize)]);
            $this->onesProb = $la->ones($p);
        }
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
            $states = $la->squeeze($states,$axis=-1);
            //$masks = $la->gather($this->masks,$states,$axis=null);
            $masks = $la->gatherb($this->masks,$states);
            $la->multiply($masks,$values);
            $la->nan2num($values,-INF);
        }
        return $values;
    }

    public function sample(NDArray $states) : NDArray
    {
        $la = $this->la;
        if($this->masks) {
            $obs = $la->squeeze($states,$axis=-1);
            //$prob = $la->gather($this->probabilities,$obs,$axis=null);
            $prob = $la->gatherb($this->probabilities,$obs);
            $actions = $this->randomCategorical($prob,1);
        } else {
            ////$values = $la->repeat($this->onesProb,count($states),$axis=0);
            $actions = $this->randomCategorical($this->onesProb,count($states));
            $actions = $la->expandDims($la->squeeze($actions),$axis=1);
        }
        return $actions;
    }

    public function __clone()
    {
        parent::__clone();
        if($this->probabilities) {
            //$this->thresholds = clone $this->thresholds;
            $this->probabilities = clone $this->probabilities;
        }
        if($this->masks) {
            $this->masks = clone $this->masks;
        }
    }
}