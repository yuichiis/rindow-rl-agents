<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Model\AbstractModel;

class GSDENoise extends AbstractModel
{
    protected Builder $nn;
    protected object $g;
    protected object $la;
    protected int $actionDim;
    protected int $featureDim;
    protected Variable $logStd;
    protected NDArray $E;
    protected int $stepCount;

    public function __construct(
        Builder $nn,
        int $actionDim,
        int $featureDim,
        ?float $logStdInit=null,
        ?string $name=null,
        )
    {
        $logStdInit ??= -0.5;

        parent::__construct($nn);
        $g = $nn->gradient();
        $la = $nn->backend()->primaryLA();
        $this->la = $la;
        $this->g = $g;
        $this->actionDim = $actionDim;
        $this->featureDim = $featureDim;
        # trainable per-action log std
        $this->logStd = $g->Variable($la->fill($logStdInit, $la->alloc([$actionDim])),trainable:true,name:$name);
        # exploration matrix E stored as buffer (resampled in-place)
        $this->E = $la->randomNormal([$featureDim, $actionDim],mean:0.0,scale:1.0);
        $this->stepCount = 0;
    }

    public function logStd(): Variable
    {
        return $this->logStd;
    }

    public function sigma(): Variable
    {
        $g = $this->g;
        return $g->exp($this->logStd);
    }

    public function resetNoise() : void
    {
        //$this->la->randomNormal([$this->featureDim, $this->actionDim],mean:0.0,scale:1.0,output:$this->E);
        $this->la->randomNormal([$this->featureDim, $this->actionDim],mean:0.0,scale:1.0,output:$this->E);
        //echo "resetNoise\n";
    }

    public function maybeResample(int $sdeSampleFreq) : void
    {
        if($sdeSampleFreq == -1) {
            return;
        }
        if(($this->stepCount % max(1, $sdeSampleFreq)) == 0) {
            $this->resetNoise();
        }
        $this->stepCount++;
    }

    public function sample(
        Variable $mu,               // (batchs, action_dim)
        Variable $phi,              // (batchs, feature_dim)
        ?bool $deterministic=null,
        ) : Variable
    {
        return $this->call($mu,$phi,deterministic:$deterministic);
    }

    public function call(
        Variable $mu,               // (batchs, action_dim)
        Variable $phi,              // (batchs, feature_dim)
        ?bool $deterministic=null,
        ) : Variable
    {
        $g = $this->g;
        $la = $this->la;
        $deterministic ??= false;
        if($deterministic) {
            $action = $mu;
        } else {
            //echo $this->la->toString($g->reduceSum($this->E))."\n";
            //echo $la->shapeToString($phi->shape()).":";
            //echo sprintf("min=%5.3f",$la->min($phi));
            //echo sprintf(",max=%5.3f",$la->max($phi))."\n";
            //$e = $this->E;
            //echo $la->shapeToString($e->shape()).":";
            //echo sprintf("min=%5.3f",$la->min($e));
            //echo sprintf(",max=%5.3f",$la->max($e))."\n";
            $noiseTerm = $g->matmul($phi,$this->E);
            //echo "noise:".$la->shapeToString($noiseTerm->shape()).":";
            //echo sprintf("min=%5.3f",$la->min($noiseTerm));
            //echo sprintf(",max=%5.3f",$la->max($noiseTerm))."\n";
            //echo "mu   :".$la->shapeToString($mu->shape()).":";
            //echo sprintf("min=%5.3f",$la->min($mu));
            //echo sprintf(",max=%5.3f",$la->max($mu))."\n";
            $action = $g->add($mu, $g->mul($noiseTerm, $this->sigma()));
            //echo "actio:".$la->shapeToString($action->shape()).":";
            //echo sprintf("min=%5.3f",$la->min($action));
            //echo sprintf(",max=%5.3f",$la->max($action))."\n";
        }
        return $action;
    }
}
