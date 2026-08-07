<?php
namespace Rindow\RL\Agents\Distribution;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use InvalidArgumentException;
use Rindow\RL\Agents\Distribution;

class Categorical implements Distribution
{
    protected Builder $nn;
    protected object $g;
    protected Variable $logits;
    protected Variable $logProbsAll;
    protected array $batchShape;
    protected int $numActions;
    protected int $batchDims;

    public function __construct(
        Builder $nn,
        ?Variable $logits=null,
        ?Variable $probs=null,
        )
    {
        $this->nn = $nn;
        $g = $nn->gradient();
        $this->g = $g;
        if($logits===null) {
            if($probs===null) {
                throw new InvalidArgumentException("Either logits or prob is required.");
            }
            $logits = $g->log($probs);
        } else {
            if($probs!==null) {
                throw new InvalidArgumentException("Either logits or prob can be specified.");
            }
        }
        $logProbsAll = $g->logSoftmax($logits); // (batchsize,numActions) : float32
        $batchShape = $logits->shape();
        $numActions = array_pop($batchShape);
        $batchDims = count($batchShape);

        $this->logits = $logits;
        $this->logProbsAll = $logProbsAll;
        $this->batchShape = $batchShape;
        $this->numActions = $numActions;
        $this->batchDims = $batchDims;

    }

    public function logProb(Variable $value) : Variable
    {
        $g = $this->g;
        if($this->batchShape!==$value->shape()) {
            $shapeString = $la->shapeToString($batchShape);
            $valueShape = $la->shapeToString($value->shape());
            throw new InvalidArgumentException("The shape of the value does not match. shape must be {$shapeString}. {$valueShape} given.");
        }
        $logProb = $g->gather($this->logProbsAll, $value, batchDims:$this->batchDims); // (batchsize) : float32
        return $logProb;
    }

    public function entropy() : Variable
    {
        $g = $this->g;
        $logits = $this->logits;
        $logits = $this->logits;
        if($logits->ndim()<=2) {
            $logits = $g->expandDims($logits,axis:0);
        }
        $probs = $g->softmax($logits);  // (batchsize,numActions)
        if($this->logits->ndim()!=$logits->ndim()) {
            $probs = $g->squeeze($probs,axis:0);
        }
        $entropy = $g->scale(-1,$g->reduceSum($g->mul($probs, $this->logProbsAll), axis:-1));
        return $entropy;
    }

    public function sample(?array $batchShape=null) : Variable
    {
        $g = $this->g;
        $sample = $g->randomCategorical($this->logits, batchShape:$batchShape);
        return $sample;
    }
}
