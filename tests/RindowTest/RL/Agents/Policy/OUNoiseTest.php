<?php
namespace RindowTest\RL\Agents\Policy\OUNoiseTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Policy\OUNoise;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use LogicException;
use InvalidArgumentException;

class TestQPolicy implements QPolicy
{
    public function __construct($la)
    {
        $this->la = $la;
    }

    public function obsSize()
    {
        return [1];
    }

    public function numActions() : int
    {
        return 1;
    }

    /**
    * @param NDArray $state
    * @return NDArray $qValues
    */
    public function getQValues($state) : NDArray
    {
        return $this->la->array($state);
    }

    public function sample($state)
    {
        return 1;
    }
}

class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newLa($mo)
    {
        return $mo->la();
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $times = 1000;
        $mean = $la->array([0.0]);
        $std_dev = $la->array([0.1]);
        $lower_bound = $la->array([-1.0]);
        $upper_bound = $la->array([ 1.0]);
        $qpolicy = new TestQPolicy($la);
        $policy = new OUNoise(
            $la, $qpolicy,
            $mean, $std_dev, $lower_bound, $upper_bound
        );

        $actions = [];
        $v = ($upper_bound[0]-$lower_bound[0])/$times;
        $c = $lower_bound[0];
        $prods = [];
        for($i=0;$i<$times;$i++) {
            $q = $c+$v*$i;
            $actions[] = $policy->action([$q],true)[0];
            $prods[] = $policy->action([$q],false)[0];
        }
        $actions = $la->array($actions);
        $prods = $la->array($prods);
        $plt->plot($actions);
        $plt->plot($prods);
        $plt->plot($la->fill($upper_bound[0],$la->alloc([$times])));
        $plt->plot($la->fill($mean[0],$la->alloc([$times])));
        $plt->plot($la->fill($lower_bound[0],$la->alloc([$times])));
        $plt->legend(['action','prod','upper','mean','lower']);
        $plt->title('OUNoise');
        $plt->show();
        $this->assertTrue(true);
    }
}