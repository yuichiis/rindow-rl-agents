<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

trait Random
{
    public function generateProbabilities($rules)
    {
        $la = $this->la;
        if($rules->ndim()!=2) {
            throw new InvalidArgumentException('rules must be 2D NDArray');
        }
        // p[i] = exp(rules[i])/sum(exp(rules[i]))
        $prob = $la->exp($la->copy($rules));
        $prob = $la->nan2num($prob);
        $sum = $la->reduceSum($prob,$axis=1);
        $la->multiply($la->reciprocal($sum),$prob,$trans=true);
        return $prob;
    }

    public function generateThresholds(NDArray $probs) : NDArray
    {
        $la = $this->la;
        if($probs->ndim()!=2) {
            throw new InvalidArgumentException('probabilities must be 2D NDArray');
        }
        [$m,$n] = $probs->shape();
        $p2 = $la->zeros($la->alloc([$m,$n-1],$probs->dtype()));
        foreach ($probs as $key => $value) {
            $la->cumsum($value[[0,$n-2]],null,null,$p2[$key]);
        }
        return $p2;
    }

    public function randomChoice(
        NDArray $probabilities,
        bool $isThresholds=null,
        bool $expandDims=null) : int
    {
        $la = $this->la;
        $thresholds = $probabilities;
        if(!$isThresholds) {
            if($expandDims) {
                $thresholds = $la->expandDims($thresholds,$axis=0);
            }
            $thresholds = $this->generateThresholds($thresholds);
            if($expandDims) {
                $thresholds = $la->squeeze($thresholds,$axis=0);
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

    public function randomCategorical(
        NDArray $probabilities,
        int $numSamples,
        $dtype=null,
    ) : NDArray
    {
        $la = $this->la;
        if($probabilities->ndim()!=2) {
            throw new InvalidArgumentException('probabilities must be 2D NDArray');
        }
        [$count,$size] = $probabilities->shape();
        if($dtype===null) {
            $dtype = NDArray::uint32;
        }
        $randints = $la->alloc([$count,$numSamples],$dtype);
        foreach ($probabilities as $key => $p) {
            $thresholds = $la->cumsum($p);
            $high = $thresholds[$size-1];
            $rand = $la->randomUniform([$numSamples],$low=0.0,$high,$thresholds->dtype());
            $la->searchsorted($thresholds,$rand,true,null,$randints[$key]);
        }
        return $randints;
    }
}
