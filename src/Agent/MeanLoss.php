<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use DomainException;

class MeanLoss extends AbstractGradient implements Loss
{
    protected $backend;
    protected $trues;
    protected $predicts;

    public function __construct($backend,array $options=null)
    {
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $K = $backend;
    }

    public function getConfig() : array
    {
        return [
        ];
    }

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        // trues : N/A
        $K = $this->backend;
        $this->predicts = $predicts;
        $this->loss = $K->mean($predicts);
        return $this->loss;
    }

    public function backward(array $dOutputs) : array
    {
        $K = $this->backend;
        $n = $this->predicts->size();
        $dInputs = $K->fill(1/$n,$K->alloc($this->predicts->shape(),$this->predicts->dtype()));
        return [$dInputs];
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        // trues : N/A
        $K = $this->backend;
        // calc accuracy
        return $K->mean($predicts);
    }
}
