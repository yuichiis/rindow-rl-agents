<?php
namespace Rindow\RL\Agents\Agent;

use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class DiscreteTrainingModel extends AbstractModel
{
    protected $la;
    protected $qNetwork;
    protected $idxInput;
    protected $gather;

    public function __construct(
        $la,$builder,$qNetwork)
    {
        parent::__construct($builder->backend(),$builder);
        $this->la = $la;
        $this->qNetwork = $qNetwork;
        $this->idxInput = $builder->layers->Input(['shape'=>[]]);
        $this->gather = $builder->layers->Gather(['axis'=>-1]);
    }

    public function compileQModel(
        object $loss=null, array $lossOpts=null,
        object $optimizer=null, array $optimizerOpts=null)
    {
        $nn = $this->builder;
        if($lossOpts===null) {
            $lossOpts=[];
        }
        if($optimizerOpts===null) {
            $optimizerOpts=[];
        }
        if($loss===null) {
            $loss=$nn->losses->Huber($lossOpts);
        }
        if($optimizer===null) {
            $optimizer=$nn->optimizers->Adam($optimizerOpts);
        }
        $loss = clone $loss;
        $optimizer = clone $optimizer;
        $nn = $this->builder;
        $this->compile([
            'loss'=>clone $loss,
            'optimizer'=>$optimizer,
            'numInputs'=>2]);
    }

    public function qNetwork()
    {
        return $this->qNetwork;
    }

    protected function call($inputs,$training)
    {
        if(!is_array($inputs)) {
            throw new InvalidArgumentException('inputs must be set of states and actions');
        }
        [$states,$actions] = $inputs;
        $actions = $this->idxInput->forward($actions,$training);
        $qValues = $this->qNetwork->forward($states,$training);
        $qValues = $this->gather->forward([$qValues,$actions],$training);
        return $qValues;
    }

    public function copyWeights($sourceNetwork,float $tau=null) : void
    {
        $K = $this->backend;
        if($tau===null) {
            $tau = 1.0;
        }
        $source = $sourceNetwork->params();
        $target = $this->params();
        $tTau = (1.0-$tau);
        foreach (array_map(null,$source,$target) as $p) {
            [$srcParam,$targParam] = $p;
            $K->update_scale($targParam,$tTau);
            $K->update_add($targParam,$srcParam,$tau);
        }
    }
}
