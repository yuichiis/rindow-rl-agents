<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Rindow\RL\Agents\Estimator\AbstractEstimatorNetwork;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Builder\Builder;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class QNetwork extends AbstractEstimatorNetwork
{
    protected object $la;
    protected int $numActions;
    protected Model $qmodel;
    //protected $thresholds;
    protected ?NDArray $probabilities;
    protected ?NDArray $masks;
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
        //?NDArray $rules=null,
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
                $activation,$kernelInitializer);
        }
        $this->qmodel = $model;
        $this->mo = $mo;
        //$this->initializeRules($rules,[$numActions]);
        //if($rules) {
        //    if($rules->shape()[1]!=$numActions) {
        //        throw new InvalidArgumentException('The rules must match numActions');
        //    }
        //    $p = $this->generateProbabilities($rules);
        //    $this->probabilities = $p;
        //    //$this->thresholds = $this->generateThresholds($p);
        //    $this->masks = $rules;
        //} else {
        //    $p = $la->alloc([1,$numActions]);
        //    $this->onesProb = $la->ones($p);
        //}
    }

    public function model() : Model
    {
        return $this->qmodel;
    }

    protected function buildModel(
        array $stateShape,
        int $numActions,
        ?array $convLayers=null,
        ?string $convType=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
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

    protected function call(NDArray $inputs, mixed $training) : NDArray
    {
        $outputs = $this->qmodel->forward($inputs,$training);
        return $outputs;
    }

    //public function getActionValues($states) : NDArray
    //{
    //    $la = $this->la;
    //    $origStates = $states;
    //    if($states instanceof NDArray) {
    //        $states = $la->expandDims($states,$axis=0);
    //    } else {
    //        $states = $la->array([[$states]]);
    //    }
    //    $values = $this->predict($states);
    //    $values = $la->squeeze($values,axis:0);
    //    if($this->masks) {
    //        $states = $origStates;
    //        if($states instanceof NDArray) {
    //            $states = (int)$states[0];
    //        }
    //        $la->multiply($this->masks[$states],$values);
    //        $la->nan2num($values,-INF);
    //    }
    //    return $values;
    //}

    /**
    * $states : (batches,...stateShape) typeof int32 or float32
    * $actionValues : (batches,...numActions)  typeof float32
    */
    public function getActionValues(NDArray $states) : NDArray
    {
        $la = $this->la;
        $orgStates = $states;
        if($la->isInt($states)) {
            $states = $la->astype($states,NDArray::float32);
        }
        $values = $this->predict($states);
        //if($this->masks) {
        //    $states = $orgStates;
        //    $states = $la->squeeze($states,axis:-1);
        //    if($states->dtype()!==NDArray::int32) {
        //        $states = $la->astype($states,NDArray::int32);
        //    }
        //    //$mask = $la->gather($this->masks,$states,$axis=null);
        //    $mask = $la->gatherb($this->masks,$states);
        //    $la->multiply($mask,$values);
        //    $la->nan2num($values,-INF);
        //}
        return $values;
    }

    public function __clone()
    {
        parent::__clone();
        //if($this->probabilities) {
        //    //$this->thresholds = clone $this->thresholds;
        //    $this->probabilities = clone $this->probabilities;
        //}
        //if($this->masks) {
        //    $this->masks = clone $this->masks;
        //}
    }
}
