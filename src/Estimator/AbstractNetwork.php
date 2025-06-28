<?php
namespace Rindow\RL\Agents\Estimator;

use Rindow\RL\Agents\Network;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Builder\Builder;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;

abstract class AbstractNetwork extends AbstractModel implements Network
{
    protected object $la;
    protected array $stateShape;

    public function __construct(
        Builder $builder,
        array $stateShape,
    )
    {
        parent::__construct($builder);
        $this->la = $builder->backend()->primaryLA();
        $this->stateShape = $stateShape;
    }

    public function stateShape() : array
    {
        return $this->stateShape;
    }

    public function numActions() : int
    {
        return $this->numActions;
    }

    public function builder() : object
    {
        return $this->builder;
    }

    //
    // param array $convLayers = [
    //    ...[
    //      int filters,
    //      int|array<int> kernel_size,
    //      [
    //          'strides' => int|array<int> ,
    //          'batch_norm' => [...args],
    //          'activation' => [...args],
    //          'pooling' => [...args],
    //          'global_pooling' => [...args],
    //          'dropout' => [...args],
    //      ],
    //    ],
    //];
    // param string $convType = {'1D', '2D', '3D' };
    // param array $fcLayers = [
    //     ...int units,
    // ];
    // param string $activation = 'relu';
    // param string $kernelInitializer = 'random_uniform';
    // param string $name = '';
    //
    protected function buildMlpLayers(
        array $inputSize,
        ?array $convLayers=null,
        ?string $convType=null,
        ?array $fcLayers=null,
        ?string $activation=null,
        ?string $kernelInitializer=null,
        ?string $name=null,
        ) : Model
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

    //
    // object $model
    // param int $numActions
    // param array $outputOptions = [
    //      'initializer' => [
    //          'kernelInitializer' => $initializer,
    //          'options' => ...$initializerOpts,
    //      ],
    //      'activation' => $last_activation,
    // ]
    //
    public function addOutputLayer(
        object $model,
        int $numActions,
        ?array $outputOptions,
        ) : void
    {
        $nn = $this->builder;
        // Last Output Layer
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
        $last_activation = null;
        if(isset($outputOptions['activation'])) {
            $last_activation = $outputOptions['activation'];
        }
        $model->add(
            $nn->layers->Dense($numActions, activation:$last_activation, kernel_initializer:$last_init)
        );
    }

    protected function call(NDArray $inputs, ?bool $training) : NDArray
    {
        $outputs = $this->model->forward($inputs,$training);
        return $outputs;
    }

    public function copyWeights(Network $sourceNetwork, ?float $tau=null) : void
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