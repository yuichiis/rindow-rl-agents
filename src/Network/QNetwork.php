<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class QNetwork extends AbstractNetwork implements QPolicy
{
    use Random;
    protected $la;
    protected $numActions;
    protected $qmodel;
    protected $thresholds;
    protected $masks;
    protected $mo;

    public function __construct($la,$builder,
            array $obsSize, int $numActions,
            array $convLayers=null,string $convType=null,array $fcLayers=null,
            $activation=null,$kernelInitializer=null,
            NDArray $rules=null,$model=null,$mo=null
        )
    {
        parent::__construct($builder,$obsSize);
        $this->la = $la;
        $this->numActions = $numActions;
        if($fcLayers===null) {
            $fcLayers = [32, 16];
        }
        if($model===null) {
            $model = $this->buildQModel($obsSize,$numActions,
                $convLayers,$convType,$fcLayers,
                $activation,$kernelInitializer);
        }
        $this->qmodel = $model;
        $this->mo = $mo;
        if($rules) {
            if($rules->shape()[1]!=$numActions) {
                throw new InvalidArgumentException('The rules must match numActions');
            }
            $p = $this->generateProbabilities($rules);
            $this->thresholds = $this->generateThresholds($p);
            $this->masks = $rules;
        }
    }

    public function qmodel()
    {
        return $this->qmodel;
    }

    public function numActions() : int
    {
        return $this->numActions;
    }

    protected function buildQModel($obsSize,$numActions,
        $convLayers,$convType,$fcLayers,
        $activation,$kernelInitializer)
    {
        $nn = $this->builder;
        $model = $this->buildMlpLayers(
            $obsSize,
            $convLayers,$convType,$fcLayers,
            $activation,$kernelInitializer);
        $model->add($nn->layers->Dense($numActions));
        return $model;
    }

    //public function compileQModel($learningRate=null)
    //{
    //    if($learningRate===null) {
    //        $learningRate = 1e-3;
    //    }
    //    $nn = $this->builder;
    //    $this->compile(optimizer:$nn->optimizers->Adam(lr:$learningRate),
    //                loss:$nn->losses->Huber(), metrics:['loss']);
    //}

    protected function call($inputs,$training)
    {
        $outputs = $this->qmodel->forward($inputs,$training);
        return $outputs;
    }

    public function getQValues($observation) : NDArray
    {
        $la = $this->la;
        if($observation instanceof NDArray) {
            $obs = $la->expandDims($observation,$axis=0);
        } else {
            $obs = $la->array([[$observation]]);
        }
        $values = $this->predict($obs);
        $values = $la->squeeze($values,$axis=0);
        if($this->masks) {
            if($observation instanceof NDArray) {
                $observation = (int)$observation[0];
            }
            $la->multiply($this->masks[$observation],$values);
            $la->nan2num($values,-INF);
        }
        return $values;
    }

    public function getQValuesBatch(NDArray $observations) : NDArray
    {
        $la = $this->la;
        $values = $this->predict($observations);
        if($this->masks) {
            $numClass = count($observations);
            $observations = $la->squeeze($observations,$axis=-1);
            $masks = $la->gather($this->masks,$observations,$axis=null);
            $la->multiply($masks,$values);
            $la->nan2num($values,-INF);
        }
        return $values;
    }

    public function sample($state)
    {
        if($this->thresholds) {
            if($state instanceof NDArray) {
                $state = (int)$state[0];
            }
            $action = $this->randomChoice($this->thresholds[$state]);
        } else {
            $action = mt_rand(0,$this->numActions-1);
        }
        return $action;
    }

    public function __clone()
    {
        parent::__clone();
        if($this->thresholds) {
            $this->thresholds = clone $this->thresholds;
        }
        if($this->masks) {
            $this->masks = clone $this->masks;
        }
    }
}
