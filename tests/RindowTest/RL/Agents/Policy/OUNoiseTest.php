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
    public function getQValues(NDArray $state) : NDArray
    {
        return $state;
    }

    public function sample(NDArray $state) : NDArray
    {
        throw new \Exception("ILLEGAL operation", 1);
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

    public function testSingle()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $steps = 1000;
        $mean = $la->array([0.0]);
        $std_dev = $la->array([0.1]);
        $lower_bound = $la->array([-1.0]);
        $upper_bound = $la->array([ 1.0]);
        $qpolicy = new TestQPolicy($la);
        $policy = new OUNoise(
            $la,
            $mean, $std_dev, $lower_bound, $upper_bound
        );

        $actions = [];
        $v = ($upper_bound[0]-$lower_bound[0])/$steps;
        $c = $lower_bound[0];
        $prods = [];
        for($i=0;$i<$steps;$i++) {
            $q = $c+$v*$i;
            $q = $la->array([[$q]]);
            $actions[] = $policy->action($qpolicy,$q,true)[0][0];
            $prods[]   = $policy->action($qpolicy,$q,false)[0][0];
        }
        $actions = $la->array($actions);
        $prods = $la->array($prods);
        $plt->plot($actions);
        $plt->plot($prods);
        $plt->plot($la->fill($upper_bound[0],$la->alloc([$steps])));
        $plt->plot($la->fill($mean[0],$la->alloc([$steps])));
        $plt->plot($la->fill($lower_bound[0],$la->alloc([$steps])));
        $plt->legend(['action','prod','upper','mean','lower']);
        $plt->title('OUNoise');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testAvg()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $episodes = 100;
        $steps = 200;
        $mean = $la->array([0.0]);
        $std_dev = $la->array([0.2]);
        $lower_bound = $la->array([-2.0]);
        $upper_bound = $la->array([ 2.0]);
        $qpolicy = new TestQPolicy($la);
        $policy = new OUNoise(
            $la, 
            $mean, $std_dev, $lower_bound, $upper_bound
        );

        $avg = $la->zeros($la->alloc([$steps]));
        $v = ($upper_bound[0]-$lower_bound[0])/$steps;
        $c = $lower_bound[0];
        for($j=0;$j<$episodes;$j++) {
            for($i=0;$i<$steps;$i++) {
                $q = $c+$v*$i;
                $q = $la->array([[$q]]);
                $la->axpy($policy->action($qpolicy,$q,true)[0],$avg[[$i,$i]]);
            }
        }
        $la->scal(1/$episodes,$avg);
        $prods = $mo->arange($steps,$c,$v);
        $plt->plot($avg);
        $plt->plot($prods);
        $plt->plot($la->fill($upper_bound[0],$la->alloc([$steps])));
        $plt->plot($la->fill($mean[0],$la->alloc([$steps])));
        $plt->plot($la->fill($lower_bound[0],$la->alloc([$steps])));
        $plt->legend(['avg','prod','upper','mean','lower']);
        $plt->title('OUNoise');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testParallel()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $mean = $la->array([0.0,0.0]);
        $std_dev = $la->array([0.1,0.1]);
        $lower_bound = $la->array([-1.0,-1.0]);
        $upper_bound = $la->array([ 1.0, 1.0]);
        $qpolicy = new TestQPolicy($la);
        $policy = new OUNoise(
            $la,
            $mean, $std_dev, $lower_bound, $upper_bound
        );

        $actions = [];
        $prods = [];
        $q = 0.5;
        $q = $la->array([
            [$q,$q],
            [$q,$q],
            [$q,$q],
        ]);
        $actions = $policy->action($qpolicy,$q,true);
        $this->assertEquals([3,2],$actions->shape());
        $prods   = $policy->action($qpolicy,$q,false);
        $this->assertEquals([3,2],$prods->shape());
    }

}