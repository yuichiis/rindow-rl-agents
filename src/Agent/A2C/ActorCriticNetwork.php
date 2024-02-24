<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Rindow\RL\Agents\QPolicy;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Network\AbstractNetwork;

class ActorCriticNetwork extends AbstractNetwork implements QPolicy
{
    protected $actionSize;
    protected $stateLayers;
    protected $actionLayer;
    protected $criticLayer;
    protected $thresholds;
    protected $masks;

    public function __construct($la,$builder,
        array $obsSize, array $actionSize,
        array $convLayers=null,string $convType=null,array $fcLayers=null,
        string $activation=null, string $kernelInitializer=null,
        string $actionActivation=null, string $actionKernelInitializer=null,
        string $criticKernelInitializer=null,
        )
    {
        parent::__construct($builder,$obsSize);
        $this->la = $la;
        $nn = $this->builder();
        $this->actionSize = $actionSize;

        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [16, 32];
        }
        if($activation===null) {
            $activation = 'relu';
        }
        if($actionActivation===null) {
            $actionActivation = 'softmax';
        }
        $this->stateLayers = $this->buildMlpLayers(
            $obsSize,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
            name:'State'
        );
        $actionSize = (int)array_product($actionSize);
        $this->actionLayer = $nn->layers()->Dense(
            $actionSize,
            activation:$actionActivation,
            kernel_initializer:$actionKernelInitializer,
            name:'ActionDense'
        );
        $this->criticLayer = $nn->layers()->Dense(
            1,
            kernel_initializer:$criticKernelInitializer,
            name:'CriticDense'
        );
        $this->numActions = $actionSize;
    }

    public function actionSize()
    {
        return $this->actionSize;
    }

    public function call($state_input,$training=null)
    {
        $state_out = $this->stateLayers->forward($state_input,$training);
        $action_out = $this->actionLayer->forward($state_out,$training);
        $critic_out = $this->criticLayer->forward($state_out,$training);
        return [$action_out,$critic_out];
    }

    public function getQValues($observation) : NDArray
    {
        $la = $this->la;
        if(is_array($observation)) {
            if($observation[0] instanceof NDArray) {
                $obs = $la->stack($observation,$axis=0);
            } else {
                $obs = $la->expandDims($la->array($observation),$axis=0);
            }
        } elseif($observation instanceof NDArray) {
            $obs = $la->expandDims($observation,$axis=0);
        } else {
            $obs = $la->array([[$observation]]);
        }
        [$action_out,$critic_out] = $this->forward($obs);
        if(!is_array($observation)) {
            $action_out = $la->squeeze($action_out,$axis=0);
        }
        if($this->masks) {
            if($observation instanceof NDArray) {
                $observation = (int)$observation[0];
            }
            $la->multiply($this->masks[$observation],$action_out);
            $la->nan2num($action_out,-INF);
        }
        return $action_out;
    }

    public function sample(NDArray $state) : NDArray
    {
        if($this->thresholds) {
            $actions = [];
            $states = $state;
            if(!is_array($state)) {
                $states = [$state];
            }
            foreach ($states as $obs) {
                if($obs instanceof NDArray) {
                    $obs = (int)$obs[0];
                }
                $actions[] = $this->randomChoice($this->thresholds[$obs], isThresholds:true);
            }
            if(!is_array($state)) {
                $actions = $actions[0];
            }
        } else {
            $actions = [];
            if(!is_array($state)) {
                $actions = mt_rand(0,$this->numActions-1);
            } else {
                $count = count($state);
                for($i=0;$i<$count;$i++) {
                    $actions[] = mt_rand(0,$this->numActions-1);
                }
                $actions = $this->la->array($actions,NDArray::uint32);
            }
        }
        return $actions;
    }
}
