<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;

class OUNoise implements Policy
{
    protected $la;
    protected $qPolicy;

    public function __construct(
        $la, QPolicy $qPolicy,
        array $shape, $dtype, float $damping, float $stddev)
    {
        $this->la = $la;
        $this->qPolicy = $qPolicy;
        $this->ouProcess = new ouProcess($la
            $la->zeros($la->alloc($shape,$dtype)),
            $damping,
            $stddev);
    }

    public function initialize()
    {
        $this->ouProcess->reset();
    }

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($state,int $time=null)
    {
        $qValues = $this->qPolicy->getQValues($state);
        $noise = $this->ouProcess();
        return $this->la->axpy($noise,$qValues);
    }
}
