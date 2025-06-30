<?php
namespace Rindow\RL\Agents\Agent\Ddpg;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\layer\Layer;
use Rindow\RL\Agents\Estimator\AbstractNetwork;
use InvalidArgumentException;

class CriticNetwork extends AbstractNetwork
{
    protected int $numActions;
    protected Model $stateLayers;
    protected ?Model $actionLayers;
    protected Layer $concat;
    protected Model $combinedLayers;
    protected Layer $outputDense;

    //protected $stateDense1;
    //protected $stateDense2;
    //protected $actionDense;
    //protected $concat;
    //protected $concatDense1;
    //protected $concatDense2;
    //protected $outputDense;

    public function __construct(
        Builder $builder,
        array $stateShape, int $numActions,
        ?array $staConvLayers=null, ?string $staConvType=null, ?array $staFcLayers=null,
        ?array $actLayers=null,
        ?array $comLayers=null,
        ?string $activation=null, ?string $kernelInitializer=null,
        )
    {
        parent::__construct($builder,$stateShape);
        $this->numActions = $numActions;

        if($staConvLayers===null && $staFcLayers===null) {
            $staFcLayers = [16, 32];
        }
        if($actLayers===null) {
            $actLayers = [32];
        }
        if($comLayers===null) {
            $comLayers = [256,256];
        }
        $staLast = array_key_last($staFcLayers);
        $actLast = array_key_last($actLayers);
        if($staLast===null) {
            throw new InvalidArgumentException("Must specify staFcLayers.");
        }
        if($actLast===null) {
            if($staFcLayers[$staLast]!==$numActions) {
                throw new InvalidArgumentException("the same last staFcLayers must be numActions.");
            }
        } else {
            if($staFcLayers[$staLast]!==$actLayers[$actLast]) {
                throw new InvalidArgumentException("Must be the same last staFcLayers and last actLayers.");
            }
        }
        $this->stateLayers = $this->buildMlpLayers(
            $stateShape,
            convLayers:$staConvLayers,
            convType:$staConvType,
            fcLayers:$staFcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );
        if($actLast===null) {
            $this->actionLayers = null;
        } else {
            $this->actionLayers = $this->buildMlpLayers(
                [$numActions],
                fcLayers:$actLayers,
                activation:$activation,
                kernelInitializer:$kernelInitializer
            );
        }
        $this->concat = $builder->layers->Concatenate();
        $this->combinedLayers = $this->buildMlpLayers(
            [$numActions],
            fcLayers:$comLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );
        $this->outputDense = $builder->layers->Dense($numActions);
        
        //$this->stateDense1 = $nn->layers->Dense(16, activation:'relu');
        //$this->stateDense2 = $nn->layers->Dense(32, activation:'relu');
        //$this->actionDense = $nn->layers->Dense(32, activation:'relu');
        //$this->concat = $nn->layers->Concatenate();
        //$this->concatDense1 = $nn->layers->Dense(256, activation:'relu');
        //$this->concatDense2 = $nn->layers->Dense(256, activation:'relu');
        //$this->outputDense = $nn->layers->Dense(1);
    }

    public function call(NDArray $state_input, NDArray $action_input, ?bool $training)
    {
        //$state_out = $this->stateDense1->forward($state_input,$training);
        //$state_out = $this->stateDense2->forward($state_out,$training);

        //$action_out = $this->actionDense->forward($action_input,$training);

        //$concat = $this->concat->forward([$state_out, $action_out],$training);
        //$out = $this->concatDense1->forward($concat,$training);
        //$out = $this->concatDense2->forward($out,$training);

        //$out = $this->outputDense->forward($out,$training);

        $state_out = $this->stateLayers->forward($state_input,$training);
        if($this->actionLayers) {
            $action_out = $this->actionLayers->forward($action_input,$training);
        } else {
            $action_out = $action_input;
        }
        $concat = $this->concat->forward([$state_out, $action_out],$training);
        $out = $this->combinedLayers->forward($concat,$training);
        $out = $this->outputDense->forward($out,$training);
        return $out;
    }
}
