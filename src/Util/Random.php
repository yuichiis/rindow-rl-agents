<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

trait Random
{
    ///**
    // * $rules(numStates,numAction) : 1.0 or NaN
    // * $prob(numStates,numAction)  : 0.0 <= x <= 1.0
    // */
    //public function generateProbabilities($rules)
    //{
    //    $la = $this->la;
    //    if($rules->ndim()!=2) {
    //        throw new InvalidArgumentException('rules must be 2D NDArray');
    //    }
    //    // p[i] = exp(rules[i])/sum(exp(rules[i]))
    //    $prob = $la->exp($la->copy($rules));
    //    $prob = $la->nan2num($prob);
    //    $sum = $la->reduceSum($prob,axis:1);
    //    $la->multiply($la->reciprocal($sum),$prob,$trans=true);
    //    return $prob;
    //}

    public function generateThresholds(NDArray $probs) : NDArray
    {
        $la = $this->la;
        if($probs->ndim()!=2) {
            throw new InvalidArgumentException('probabilities must be 2D NDArray');
        }
        //[$m,$n] = $probs->shape();
        //$p2 = $la->zeros($la->alloc([$m,$n-1],$probs->dtype()));
        //foreach ($probs as $key => $value) {
        //    $la->cumsum($value[R(0,$n-1)],null,null,$p2[$key]);
        //}
        //return $p2;
        return $la->cumsum($probs,axis:-1);
    }

    public function randomChoice(
        NDArray $probabilities,
        ?bool $isThresholds=null,
        ?bool $expandDims=null) : int
    {
        $la = $this->la;
        $thresholds = $probabilities;
        if(!$isThresholds) {
            if($expandDims) {
                $thresholds = $la->expandDims($thresholds,$axis=0);
            }
            $thresholds = $this->generateThresholds($thresholds);
            if($expandDims) {
                $thresholds = $la->squeeze($thresholds,axis:0);
            }
        }
        if($thresholds->ndim()!=1) {
            throw new InvalidArgumentException('thresholds must be 1D NDArray');
        }
        $max = getrandmax();
        $rand = mt_rand(0,$max)/($max+1);
        $rand = $la->array([$rand]);
        $randint = $la->searchsorted($thresholds,$rand,true);
        return $randint->toArray()[0];
    }

    /**
     * $logits : (batches,numSamples) dtype:float32. log(probabilities)
     * $randints: (batches) dtype:int32
     */
    public function randomCategorical(
        NDArray $logits,
        ?int $dtype=null,
    ) : NDArray
    {
        $la = $this->la;
        if(!$la->isFloat($logits)) {
            throw new InvalidArgumentException('logits must be float dtype.');
        }
        if($logits->ndim()!=2) {
            throw new InvalidArgumentException('logits must be 2D NDArray without numSamples.');
        }
        [$batches,$numActions] = $logits->shape();
        if($dtype===null) {
            $dtype = NDArray::int32;
        }
        #echo "logits  :".$la->toString($logits,format:'%.3f')."\n";
        $probabilities = $la->softmax($la->copy($logits));
        #echo "probab  :".$la->toString($probabilities,format:'%.3f')."\n";
        $rand = $la->randomUniform([$batches],dtype:$probabilities->dtype(),low:0.0,high:1.0);// (batches)
        $thresholds = $la->cumsum($probabilities,axis:-1);      // (batches,numActions)
        $randints = $la->searchsorted(                          // (batches)
            $thresholds,$rand,
            right:true,dtype:$dtype
        );
        return $randints;
    }
}
