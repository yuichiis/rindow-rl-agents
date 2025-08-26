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

class ActorNetwork extends AbstractNetwork implements Estimator
{
    protected object $g;
    protected int $numActions;
    protected Model $stateLayers;
    protected Layer $actionLayer;
    // protected Variable $logStd; // <<< 変更点: 状態に依存しないlogStd変数を削除
    protected Layer $logStdLayer; // <<< 変更点: 状態からlogStdを出力するレイヤーを追加
    protected Layer $criticLayer;
    protected bool $continuous;
    protected ?NDArray $action_low_bound=null;
    protected ?NDArray $action_high_bound=null;
    protected ?NDArray $actionScale = null;
    protected ?NDArray $actionShift = null;
    protected float $epsilon = 1e-6;

    public function __construct(
        Builder $builder,
        array $stateShape, int $numActions,
        ?array $convLayers=null,?string $convType=null,?array $fcLayers=null,
        ?string $activation=null, ?string $kernelInitializer=null,
        ?string $actionActivation=null,
        mixed $actionKernelInitializer=null,
        mixed $criticKernelInitializer=null,
        // <<< 変更点: logStdレイヤー用の初期化方法を追加（任意）
        mixed $logStdKernelInitializer=null,
        ?NDArray $actionMin=null, ?NDArray $actionMax=null,
        //?float $initialStd=null, // <<< 変更点: initialStdは不要になるため削除
        ?bool $continuous=null,
        )
    {
        $continuous ??= false;
        $activation ??= 'relu';
        $actionMin ??= -1.0;
        $actionMax ??= 1.0;

        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [128, 128];
        }
        // $initialStd ??= 1.0; // <<< 変更点: 削除

        parent::__construct($builder,$stateShape);
        $nn = $this->builder();
        $la = $this->la;
        $this->g = $nn->gradient();

        $this->numActions = $numActions;
        $this->continuous = $continuous;
        $this->action_low_bound = $actionMin;
        $this->action_high_bound = $actionMax;

        $this->stateLayers = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer,
            name:'State'
        );
        $actionActivation ??= 'tanh';
        $this->actionLayer = $nn->layers()->Dense(
            $numActions,
            activation:$actionActivation,
            kernel_initializer:$actionKernelInitializer,
            name:'Action'
        );
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

        $logStdKernelInitializer ??= 'zeros'; // 学習初期の標準偏差をexp(0)=1.0にするため'zeros'で初期化
        $this->logStdLayer = $nn->layers()->Dense(
            $numActions,
            // 活性化関数はなし（線形出力）
            kernel_initializer:$logStdKernelInitializer,
            name:'LogStd'
        );
        // --- ▲▲▲ 変更点 ▲▲▲ ---
        $this->numActions = $numActions;
    }

    protected function logProb(NDArray $mean, NDArray $logStd, NDArray $value) : NDArray
    {
        $g = $this->g;
        $stableStd = $g->add($g->exp($logStd),1e-8);
        $logProb = $g->sub(
            $g->sub(
                $g->scale(-0.5,$g->square($g->div($g->sub($value,$mean),$stableStd))),
                $g->log($stableStd)
            ),
            (0.5 * log(2.0 * pi()))
        );
        $logProb = $g->reduceSum($logProb, axis:-1);
        return $logProb;
    }

    public function call(
        NDArray $state_input,       // (batchSize,numState)
        mixed $training=null,       // bool
        ?bool $deterministic=null,  // bool
        ) : array
    {
        $g = $this->g;
        $add_batch_dim = false;
        if($state_input->ndim() == 1) {
            $state_input = $g->expanDims($state_input, axis:0); // if(batchSize=1) (1,numState)
            $add_batch_dim = true;
        }

        $state_out = $this->stateLayers->forward($state_input,$training);   // (batchSize,units)
        $mean = $this->actionLayer->forward($state_out,$training);          // (batchSize,numActions)
        $logStd = $this->logStdLayer->forward($state_out, $training);       // (batchSize,numActions)

        if(!$deterministic) {
            $z = $mean;     // (batchSize,numActions)
        } else {
            $z = $g->add(   // (batchSize,numActions)
                $mean,
                $g->mul(
                    $g->exp($logStd),
                    $g->randomNormLike($mean)
            )); 
        }
        $action = $g->tanh($z); // (batchSize,numActions)
        $log_prob = $g->add(    // (batchSize,numActions)
            $g->log($g->add($g->sub(1, $g->square($action)), $this->epsilon)),// (batchSize,numActions)
            $g->scale(-1,$this->logProb($mean,$logStd,$z)),                 // (batchSize)
            trans:true,
        );
        if($log_prob->ndim()>1) {
            $log_prob = $g->reduceSum($log_prob, axis:1, keepdims:true);    // (batchSize,1)
        } else {
            $log_prob = $g->reduceSum($log_prob, keepdims:true);            // (batchSize,1)
        }
        $scaled_action = $g->mul($this->action_high_bound,$action);
        if($add_batch_dim) {
            $scaled_action = $g->squeeze($scaled_action, axis:0);           // (numActions)
            $log_prob = $g->squeeze($log_prob, axis:0);                     // (1)
        }
        if($deterministic) {
            return $scaled_action;                                          // (batchSize,numActions) or no-batchsize
        }
        return [$scaled_action, $log_prob];                                 // (batchSize,numActions)(batchSize,1) or no-batchsize
    }

    /**
     * @param  NDArray $states : (batches,...StateDims) typeof int32 or float32
     * @return NDArray $actionValues : (batches,...ValueDims) typeof float32
     */
    public function getActionValues(NDArray $states,?bool $std=null) : NDArray|array
    {
        $std ??= false;
        $la = $this->la;
        if($states->ndim()<2) {
            $specs = $la->dtypeToString($states->dtype())."(".implode(',',$states->shape()).")";
            throw new InvalidArgumentException("states must be a 2-dimensional array or higher. $specs given.");
        }
        $orgStates = $states;
        if($la->isInt($states)) {
            $states = $la->astype($states,NDArray::float32);
        }

        // continuous
        [$action_out,$logStd] = $this->forward($states,false);
        if($std) {
            return [$action_out,$logStd];
        } else {
            return $action_out;
        }
    }

}