<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Util\Random;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class QNetwork extends AbstractModel implements QPolicy,Network
{
    use Random;
    protected $la;
    protected $obsSize;
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
        parent::__construct($builder->backend(),$builder);
        $this->la = $la;
        $this->obsSize = $obsSize;
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

    public function builder()
    {
        return $this->builder;
    }

    public function qmodel()
    {
        return $this->qmodel;
    }

    public function obsSize()
    {
        return $this->obsSize;
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
        if($nn===null || $fcLayers===null) {
            throw new InvalidArgumentException('You need to specify the NeuralNetworks builder and the HiddenSize');
        }
        if($convType!==null&&$convType!='1d'&&$convType!='2d') {
            throw new InvalidArgumentException('Unknown convType:'.$convType);
        }
        if($activation===null) {
            $activation='relu';
        }
        if($kernelInitializer===null) {
            $kernelInitializer = 'random_uniform';
        }
        $model = $nn->models->Sequential();
        if($convLayers&&count($convLayers)) {
            $model->add($nn->layers->Input(['shape'=>$obsSize]));
            foreach ($convLayers as $config) {
                $pooling = $batchNorm = $dropout = $activation = $globalPooling = null;
                if(isset($config[2]['pooling'])) {
                    if($convType=='1d') {
                        $pooling = $nn->layers->MaxPooling1D(...$config[2]['pooling']);
                    } else {
                        $pooling = $nn->layers->MaxPooling2D(...$config[2]['pooling']);
                    }
                    unset($config[2]['pooling']);
                }
                if(isset($config[2]['batch_norm'])) {
                    $batchNorm = $nn->layers->BatchNormalization($config[2]['batch_norm']);
                    unset($config[2]['batch_norm']);
                    if(isset($config[2]['activation'])) {
                        $activation = $config[2]['activation'];
                        unset($config[2]['activation']);
                    }
                }
                if(isset($config[2]['dropout'])) {
                    $dropout = $nn->layers->Dropout(...$config[2]['dropout']);
                    unset($config[2]['dropout']);
                }
                if(isset($config[2]['global_pooling'])) {
                    if($convType=='1d') {
                        $globalPooling = $nn->layers->GlobalAveragePooling1D(...$config[2]['global_pooling']);
                    } else {
                        $globalPooling = $nn->layers->GlobalAveragePooling2D(...$config[2]['global_pooling']);
                    }
                    unset($config[2]['global_pooling']);
                }
                if($convType=='1d') {
                    $conv = $nn->layers->Conv1D(...$config);
                } else {
                    $conv = $nn->layers->Conv2D(...$config);
                }

                $model->add($conv);
                if($batchNorm!=null) {
                    $model->add($batchNorm);
                }
                if($activation!=null) {
                    $model->add($nn->layers->Activation($activation));
                }
                if($pooling!=null) {
                    $model->add($pooling);
                }
                if($globalPooling!=null) {
                    $model->add($globalPooling);
                }
                if($dropout!=null) {
                    $model->add($dropout);
                }
            }
            $flattenOptions = [];
        } else {
            $flattenOptions = ['input_shape'=>$obsSize];
        }
        $model->add($nn->layers->Flatten($flattenOptions));
        foreach ($fcLayers as $units) {
            $model->add($nn->layers->Dense($units,
                ['activation'=>$activation,'kernel_initializer'=>$kernelInitializer]));
        }
        $model->add($nn->layers->Dense($numActions));
        return $model;
    }

    public function compileQModel($learningRate=null)
    {
        if($learningRate===null) {
            $learningRate = 1e-3;
        }
        $nn = $this->builder;
        $this->compile(['optimizer'=>$nn->optimizers->Adam(['lr'=>$learningRate]),
                    'loss'=>$nn->losses->Huber(), 'metrics'=>['loss']]);
    }

    protected function call($inputs,$training)
    {
        $outputs = $this->qmodel->forward($inputs,$training);
        return $outputs;
    }

    public function copyWeights($sourceNetwork,float $tau=null) : void
    {
        $K = $this->backend;
        if($tau===null) {
            $tau = 1.0;
        }
        $source = $sourceNetwork->params();
        $target = $this->params();
        $tTau = (1.0-$tau);
        foreach (array_map(null,$source,$target) as $p) {
            [$srcParam,$targParam] = $p;
            $K->update_scale($targParam,$tTau);
            $K->update_add($targParam,$srcParam,$tau);
            //$la->axpy($srcParam,$la->scal($tTau,$targParam),$tau);
        }
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
