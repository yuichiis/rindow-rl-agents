<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\Network;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;

abstract class AbstractNetwork extends AbstractModel implements Network
{
    protected $obsSize;

    public function __construct($builder,array $obsSize)
    {
        parent::__construct($builder->backend(),$builder);
        $this->obsSize = $obsSize;
    }

    public function obsSize()
    {
        return $this->obsSize;
    }

    public function actionSize()
    {
        throw new LogicException('Unsupported operation: actionSize');
    }

    public function numActions() : int
    {
        throw new LogicException('Unsupported operation: numActions');
    }

    public function builder()
    {
        return $this->builder;
    }

    public function getQValuesBatch(NDArray $observations) : NDArray
    {
        throw new LogicException('Unsupported operation');
    }

    protected function buildMlpLayers(
        array $inputSize,
        $convLayers=null,
        $convType=null,
        $fcLayers=null,
        $activation=null,
        $kernelInitializer=null,
        $name=null)
    {
        $nn = $this->builder;
        $name = $name ?? '';
        if($nn===null || ($fcLayers===null && $convLayers===null)) {
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
        $enableFlatten = false;
        $model = $nn->models->Sequential();
        if($convLayers&&count($convLayers)) {
            $model->add($nn->layers->Input(shape:$inputSize));
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
                    $batchNorm = $nn->layers->BatchNormalization(...$config[2]['batch_norm']);
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
                $convOpts = $config[2];
                unset($config[2]);
                if($convType=='1d') {
                    $conv = $nn->layers->Conv1D($config[0],$config[1],...$convOpts);
                } else {
                    $conv = $nn->layers->Conv2D($config[0],$config[1],...$convOpts);
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
            $enableFlatten = true;
        } else {
            if(count($inputSize)>1) {
                $flattenOptions = ['input_shape'=>$inputSize];
                $enableFlatten = true;
            }
        }
        if($enableFlatten) {
            $model->add($nn->layers->Flatten(...$flattenOptions));
        }
        $i = 0;
        foreach ($fcLayers as $units) {
            $model->add($nn->layers->Dense($units,
                activation:$activation,kernel_initializer:$kernelInitializer,name:"{$name}FcDense{$i}"));
            $i++;
        }
        return $model;
    }

    public function copyWeights($sourceNetwork,float $tau=null) : void
    {
        $K = $this->backend;
        if($tau===null) {
            $tau = 1.0;
        }
        $source = $sourceNetwork->variables();
        $target = $this->variables();
        $tTau = (1.0-$tau);
        foreach (array_map(null,$source,$target) as [$srcParam,$targParam]) {
            $K->update_scale($targParam->value(),$tTau);
            $K->update_add($targParam->value(),$srcParam->value(),$tau);
        }
    }

}