<?php
namespace Rindow\RL\Agents\Agent\Ddpg;

use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Util\Random;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;

class ActorNetwork extends AbstractEstimatorNetwork implements Estimator
{
    protected object $model;
    protected array $actionShape;
    //protected $onesProb;
    protected ?NDArray $masks=null;
    protected ?NDArray $probabilities=null;

    public function __construct($builder,
            array $stateShape,
            int|array $actionShape,
            ?array $convLayers=null,?string $convType=null,?array $fcLayers=null,
            $activation=null,$kernelInitializer=null,
            ?array $outputOptions=null,
            ?NDArray $rules=null,$model=null
        )
    {
        if(is_int($actionShape)) {
            $actionShape = [$actionShape];
        }
        parent::__construct($builder,$stateShape);
        $this->actionShape = $actionShape;

        $la = $this->la;
        if($model===null) {
            $model = $this->buildActorModel(
                $stateShape,$actionShape,
                $convLayers,$convType,$fcLayers,
                $activation,$kernelInitializer,
                $outputOptions,
            );
        }
        $this->initializeRules($rules,$actionShape);
        $this->model = $model;
    }

    protected function buildActorModel(
        $stateShape,$actionShape,
        $convLayers,$convType,$fcLayers,
        $activation,$kernelInitializer,
        $outputOptions,
        )
    {
        $nn = $this->builder;
        $K = $this->backend;

        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [32, 16];
        }

        $model = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );

        $this->addOutputLayer(
            $model,
            $actionShape,
            $outputOptions,
        );

        return $model;
    }

    public function __clone()
    {
        parent::__clone();
        if($this->probabilities) {
            //$this->thresholds = clone $this->thresholds;
            $this->probabilities = clone $this->probabilities;
        }
        if($this->masks) {
            $this->masks = clone $this->masks;
        }
    }
}