<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;

class CriticNetwork extends AbstractNetwork
{
    protected $actionSize;
    protected $stateLayers;
    protected $actionLayers;
    protected $concat;
    protected $concatLayers;
    protected $outputDense;

    //protected $stateDense1;
    //protected $stateDense2;
    //protected $actionDense;
    //protected $concat;
    //protected $concatDense1;
    //protected $concatDense2;
    //protected $outputDense;

    public function __construct($la,$builder,
        array $obsSize, array $actionSize,
        array $obsConvLayers=null,string $obsConvType=null,array $obsFcLayers=null,
        array $actConvLayers=null,string $actConvType=null,array $actFcLayers=null,
        array $conConvLayers=null,string $conConvType=null,array $conFcLayers=null,
        string $activation=null, string $kernelInitializer=null,
        )
    {
        parent::__construct($builder,$obsSize);
        $this->actionSize = $actionSize;

        if($obsConvLayers===null && $obsFcLayers===null) {
            $obsFcLayers = [16, 32];
        }
        if($actConvLayers===null && $actFcLayers===null) {
            $actFcLayers = [32];
        }
        if($conConvLayers===null && $conFcLayers===null) {
            $conFcLayers = [256,256];
        }
        $this->stateLayers = $this->buildMlpLayers(
            $obsSize,
            convLayers:$obsConvLayers,
            convType:$obsConvType,
            fcLayers:$obsFcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );
        $this->actionLayers = $this->buildMlpLayers(
            $actionSize,
            convLayers:$actConvLayers,
            convType:$actConvType,
            fcLayers:$actFcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );
        $this->concatLayers = $this->buildMlpLayers(
            $actionSize,
            convLayers:$conConvLayers,
            convType:$conConvType,
            fcLayers:$conFcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );
        $this->concat = $builder->layers->Concatenate();
        $this->outputDense = $builder->layers->Dense(1);
        
        //$this->stateDense1 = $nn->layers->Dense(16, activation:'relu');
        //$this->stateDense2 = $nn->layers->Dense(32, activation:'relu');
        //$this->actionDense = $nn->layers->Dense(32, activation:'relu');
        //$this->concat = $nn->layers->Concatenate();
        //$this->concatDense1 = $nn->layers->Dense(256, activation:'relu');
        //$this->concatDense2 = $nn->layers->Dense(256, activation:'relu');
        //$this->outputDense = $nn->layers->Dense(1);
    }

    public function actionSize()
    {
        return $this->actionSize;
    }

    public function call($state_input,$action_input,$training)
    {
        //$state_out = $this->stateDense1->forward($state_input,$training);
        //$state_out = $this->stateDense2->forward($state_out,$training);

        //$action_out = $this->actionDense->forward($action_input,$training);

        //$concat = $this->concat->forward([$state_out, $action_out],$training);
        //$out = $this->concatDense1->forward($concat,$training);
        //$out = $this->concatDense2->forward($out,$training);

        //$out = $this->outputDense->forward($out,$training);

        $state_out = $this->stateLayers->forward($state_input,$training);
        $action_out = $this->actionLayers->forward($action_input,$training);
        $concat = $this->concat->forward([$state_out, $action_out],$training);
        $out = $this->concatLayers->forward($concat,$training);
        $out = $this->outputDense->forward($out,$training);
        return $out;
    }
}
