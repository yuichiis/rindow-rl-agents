<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Estimator\AbstractNetwork;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Builder\Builder;
use InvalidArgumentException;
use LogicException;

class ActorCriticNetwork extends AbstractNetwork implements Estimator
{
    protected int $numActions;
    protected Model $stateLayers;
    protected Layer $actionLayer;
    protected Variable $logStd;
    protected Layer $criticLayer;
    protected bool $continuous;
    //protected ?NDArray $actionMin=null;
    //protected ?NDArray $actionMax=null;
    protected ?NDArray $actionScale = null;
    protected ?NDArray $actionShift = null;

    public function __construct(
        object $la,
        Builder $builder,
        array $stateShape, int $numActions,
        ?array $convLayers=null,?string $convType=null,?array $fcLayers=null,
        ?string $activation=null, ?string $kernelInitializer=null,
        ?string $actionActivation=null,
        mixed $actionKernelInitializer=null,
        mixed $criticKernelInitializer=null,
        ?NDArray $actionMin=null, ?NDArray $actionMax=null,
        ?float $initialStd=null,
        ?bool $continuous=null,
        )
    {
        $continuous ??= false;
        //if(!$continuous) {
            $activation ??= 'relu';
        //} else {
        //    $kernelInitializer ??= 'he_uniform';
        //    $activation ??= 'tanh';
        //    $actionActivation ??= 'tanh';
        //}
        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [64, 64];
        }
        $initialStd ??= 1.0;

        parent::__construct($builder,$stateShape);
        $this->la = $la;
        $nn = $this->builder();

        $this->numActions = $numActions;
        $this->continuous = $continuous;

        $this->stateLayers = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
            name:'State'
        );
        if($continuous) {
            $actionActivation ??= 'tanh';
        }
        $this->actionLayer = $nn->layers()->Dense(
            $numActions,
            activation:$actionActivation,
            kernel_initializer:$actionKernelInitializer,
            name:'Action'
        );
        if($continuous) {
            // actionMin と actionMax から scale と shift を計算
            if($actionMin!==null && $actionMax!==null) {
                // scale = (max - min) / 2.0
                $scale = $la->scal(0.5, $la->axpy($actionMin, $la->copy($actionMax), -1.0));
                // shift = (max + min) / 2.0
                $shift = $la->scal(0.5, $la->axpy($actionMin, $la->copy($actionMax), 1.0));
                
                // 計算グラフ内で使えるように定数として保持
                $this->actionScale = $nn->gradient()->constant($scale);
                $this->actionShift = $nn->gradient()->constant($shift);
            }
            $logStd = log($initialStd);
            $this->logStd = $nn->gradient()->Variable(
                $la->fill($logStd,$la->alloc([$numActions])),
                name:'logStd',
                trainable:true,
            );
        }
        $this->criticLayer = $nn->layers()->Dense(
            1,
            kernel_initializer:$criticKernelInitializer,
            name:'Critic'
        );
        $this->numActions = $numActions;
    }

    public function call(NDArray $state_input, mixed $training=null) : array
    {
        $state_out = $this->stateLayers->forward($state_input,$training);
        $action_out = $this->actionLayer->forward($state_out,$training);
        $critic_out = $this->criticLayer->forward($state_out,$training);

        if(!$this->continuous) {
            return [            // discrete outputs
                $action_out,    // mean
                $critic_out
            ];
        } else {
            $g = $this->builder->gradient();
            if($this->actionScale && $this->actionShift) {
                // action_out = action_out * scale + shift
                $action_out = $g->add(
                    $g->mul($action_out, $this->actionScale),
                    $this->actionShift
                );
            }
            //$logstd_out = $g->clipByValue($this->logStd,-20,2);
            $logstd_out = $this->logStd;
            return [            // continuous outputs
                $action_out,    // mu acions (batchsize,numActions)
                $critic_out,    // values    (batchsize,1)
                $logstd_out, // log(std)  (numActions)
            ];
        }
    }

    /**
     * @param  NDArray $states : (batches,...StateDims) typeof int32 or float32
     * @return NDArray $actionValues : (batches,...ValueDims) typeof float32
     */
    public function getActionValues(NDArray $states) : NDArray
    {
        $la = $this->la;
        if($states->ndim()<2) {
            $specs = $la->dtypeToString($states->dtype())."(".implode(',',$states->shape()).")";
            throw new InvalidArgumentException("states must be a 2-dimensional array or higher. $specs given.");
        }
        $orgStates = $states;
        if($la->isInt($states)) {
            $states = $la->astype($states,NDArray::float32);
        }

        if(!$this->continuous) {
            [$action_out,$critic_out] = $this->forward($states,false);
        } else {
            [$action_out,$critic_out, $logStd] = $this->forward($states,false);
        }
        return $action_out;
    }

    public function getLogStd() : NDArray
    {
        if(!$this->continuous) {
            throw new LogicException("It can't get LogStd if this model is for discrete actions.");
        }
        //$g = $this->builder->gradient();
        //$logstd_out = $g->clipByValue($this->logStd,-20,2);
        $logstd_out = $this->logStd;
        return $logstd_out;
    }
}
