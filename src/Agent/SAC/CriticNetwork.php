<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Estimator\AbstractNetwork;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Builder\Builder;
use InvalidArgumentException;
use LogicException;

class CriticNetwork extends AbstractNetwork
{
    protected int $numActions;
    protected Model $q1StateLayers;
    protected Layer $q1CriticLayer;
    protected Model $q2StateLayers;
    protected Layer $q2CriticLayer;
    protected object $g;

    public function __construct(
        Builder $builder,
        array $stateShape, int $numActions,
        ?array $convLayers=null,?string $convType=null,?array $fcLayers=null,
        ?string $activation=null, ?string $kernelInitializer=null,
        mixed $criticKernelInitializer=null,
        )
    {
        $activation ??= 'relu';
        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [128, 128];
        }
        // $initialStd ??= 1.0; // <<< 変更点: 削除

        parent::__construct($builder,$stateShape);
        $nn = $this->builder();
        $la = $this->la;
        $this->g = $nn->gradient();

        $this->numActions = $numActions;
        $this->q1StateLayers = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
            name:'Q1State'
        );
        $this->q1CriticLayer = $nn->layers()->Dense(
            1,
            kernel_initializer:$criticKernelInitializer,
            name:'Q1Critic'
        );

        $this->q2StateLayers = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
            name:'Q2State'
        );
        $this->q2CriticLayer = $nn->layers()->Dense(
            1,
            kernel_initializer:$criticKernelInitializer,
            name:'Q2Critic'
        );
        $this->numActions = $numActions;
    }

    public function call(NDArray $state, NDArray $action, mixed $training=null) : array
    {
        $g = $this->g;
        $sa = $g->concat([$state, $action], axis:1);
        $q1 = $this->q1StateLayers->forward($sa,$training);
        $q1 = $this->q1CriticLayer->forward($q1,$training);
        $q2 = $this->q2StateLayers->forward($sa,$training);
        $q2 = $this->q2CriticLayer->forward($q2,$training);
        return [$q1,$q2];
    }
}
