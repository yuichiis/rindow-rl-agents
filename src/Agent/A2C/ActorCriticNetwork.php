<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Rindow\RL\Agents\Estimator;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Network\AbstractNetwork;

class ActorCriticNetwork extends AbstractNetwork implements Estimator
{
    protected $numActions;
    protected $stateLayers;
    protected $actionLayer;
    protected $criticLayer;
    protected $thresholds;
    protected $masks;

    public function __construct($la,$builder,
        array $stateShape, int $numActions,
        array $convLayers=null,string $convType=null,array $fcLayers=null,
        string $activation=null, string $kernelInitializer=null,
        string $actionActivation=null, string $actionKernelInitializer=null,
        string $criticKernelInitializer=null,
        )
    {
        parent::__construct($builder,$stateShape);
        $this->la = $la;
        $nn = $this->builder();
        $this->numActions = $numActions;

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
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
            name:'State'
        );
        $this->actionLayer = $nn->layers()->Dense(
            $numActions,
            activation:$actionActivation,
            kernel_initializer:$actionKernelInitializer,
            name:'ActionDense'
        );
        $this->criticLayer = $nn->layers()->Dense(
            1,
            kernel_initializer:$criticKernelInitializer,
            name:'CriticDense'
        );
        $this->numActions = $numActions;
    }

    public function call($state_input,$training=null)
    {
        $state_out = $this->stateLayers->forward($state_input,$training);
        $action_out = $this->actionLayer->forward($state_out,$training);
        $critic_out = $this->criticLayer->forward($state_out,$training);
        return [$action_out,$critic_out];
    }

    public function getActionValues($states) : NDArray
    {
        $la = $this->la;
        $orgStates = $states; 
        if(is_array($states)) {
            if($states[0] instanceof NDArray) {
                $states = $la->stack($states,$axis=0);
            } else {
                $states = $la->expandDims($la->array($states),$axis=0);
            }
        } elseif($states instanceof NDArray) {
            $states = $la->expandDims($states,$axis=0);
        } else {
            $states = $la->array([[$states]]);
        }
        [$action_out,$critic_out] = $this->forward($states);
        if(!is_array($orgStates)) {
            $action_out = $la->squeeze($action_out,axis:0);
        }
        if($this->masks) {
            $states = $orgStates;
            if($states instanceof NDArray) {
                $states = (int)$states[0];
            }
            $la->multiply($this->masks[$states],$action_out);
            $la->nan2num($action_out,-INF);
        }
        return $action_out;
    }
}
