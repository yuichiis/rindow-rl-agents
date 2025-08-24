<?php
namespace RindowTest\RL\Agents\Policy\OUNoiseTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Policy\OUNoise;
use Rindow\RL\Agents\ReplayBuffer\QueueBuffer;
use LogicException;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

class TestEstimator implements Estimator
{
    protected $la;

    public function __construct($la)
    {
        $this->la = $la;
    }

    public function stateShape() : array
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
    public function getActionValues(NDArray $state,?bool $std=null) : NDArray|array
    {
        return $state;
    }

    //public function probabilities(NDArray $state) : NDArray
    //{
    //    throw new \Exception("ILLEGAL operation", 1);
    //}
}

class OUNoiseTest extends TestCase
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
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
            'renderer.execBackground' => true,
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
        $estimator = new TestEstimator($la);
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
            $actions[] = $policy->actions($estimator,$q,training:true,masks:null)[0][0];
            $prods[]   = $policy->actions($estimator,$q,training:false,masks:null)[0][0];
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

        $episodes = 50;
        $steps = 100;
        $mean = $la->array([0.0]);
        $std_dev = $la->array([0.2]);
        $lower_bound = $la->array([-2.0]);
        $upper_bound = $la->array([ 2.0]);
        $estimator = new TestEstimator($la);
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
                $la->axpy($policy->actions($estimator,$q,training:true,masks:null)[0],$avg[R($i,$i+1)]);
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
        $estimator = new TestEstimator($la);
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
        $actions = $policy->actions($estimator,$q,training:true,masks:null);
        $this->assertEquals([3,2],$actions->shape());
        $prods   = $policy->actions($estimator,$q,training:false,masks:null);
        $this->assertEquals([3,2],$prods->shape());
    }

}