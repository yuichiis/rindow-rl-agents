<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Network\AbstractNetwork;

class ActorCriticNetwork extends AbstractNetwork
{
    protected $actionSize;
    protected $stateLayers;
    protected $actionLayer;
    protected $criticLayer;

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
        if($actionActivation===null) {
            $actionActivation = 'softmax';
        }
        $this->stateLayers = $this->buildMlpLayers(
            $obsSize,
            convLayers:$obsConvLayers,
            convType:$obsConvType,
            fcLayers:$obsFcLayers,
            activation:$obsActivation,
            kernelInitializer:$kernelInitializer
        );
        $this->actionLayer = $this->nn->Dense(
            $actionSize,
            activation:$actionActivation,
            kernelInitializer:$actionKernelInitializer
        );
        $this->criticLayer = $this->nn->Dense(
            1,
            kernelInitializer:$criticKernelInitializer
        );
        
    }

    public function actionSize()
    {
        return $this->actionSize;
    }

    public function call($state_input,$training)
    {
        $state_out = $this->stateLayers->forward($state_input,$training);
        $action_out = $this->actionLayer->forward($state_out,$training);
        $critic_out = $this->criticLayer->forward($state_out,$training);
        return [$action_out,$critic_out];
    }
}
