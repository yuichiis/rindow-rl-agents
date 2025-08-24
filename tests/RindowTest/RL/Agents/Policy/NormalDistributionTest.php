<?php
namespace RindowTest\RL\Agents\Policy\NormalDistributionTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Policy\NormalDistribution;
use Rindow\RL\Agents\ReplayBuffer\QueueBuffer;
use LogicException;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

class TestEstimator implements Estimator
{
    protected object $la;
    protected int $numActions;
    protected NDArray $logStd;

    public function __construct($la,$numActions)
    {
        $this->la = $la;
        $this->numActions = $numActions;
        $std = 0.2;
        $this->logStd = $la->log($la->fill($std,$la->alloc([$numActions])));
    }

    public function stateShape() : array
    {
        return [1];
    }

    public function numActions() : int
    {
        return $this->numActions;
    }

    /**
    * @param NDArray $state
    * @return NDArray $qValues
    */
    public function getActionValues(NDArray $state,?bool $std=null) : NDArray|array
    {
        if($std) {
            $la = $this->la;
            $shape = $state->shape();
            array_shift($shape);
            return [$state,$la->ones($la->alloc($shape))];
        } else {
            return $state;
        }
    }

    //public function probabilities(NDArray $state) : NDArray
    //{
    //    throw new \Exception("ILLEGAL operation", 1);
    //}
}

class NormalDistributionTest extends TestCase
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
        $estimator = new TestEstimator($la,1);
        $policy = new NormalDistribution(
            $la,
            min:$lower_bound, max:$upper_bound
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
        $plt->title('Normal');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testMulti()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $steps = 1000;
        $mean = $la->array([0.0]);
        $std_dev = $la->array([0.1]);
        $lower_bound = $la->array([-1.0,0.0]);
        $upper_bound = $la->array([ 0.0,1.0]);
        $estimator = new TestEstimator($la,2);
        $policy = new NormalDistribution(
            $la,
            min:$lower_bound, max:$upper_bound
        );

        $actions0 = [];
        $actions1 = [];
        $v = ($upper_bound[0]-$lower_bound[0])/$steps;
        $c = $lower_bound[0];
        $prods0 = [];
        $prods1 = [];
        for($i=0;$i<$steps;$i++) {
            $q = $c+$v*$i;
            $q = $la->array([[$q,$q+1]]);
            $action = $policy->actions($estimator,$q,training:true,masks:null)[0];
            $actions0[] = $action[0];
            $actions1[] = $action[1];
            $prod = $policy->actions($estimator,$q,training:false,masks:null)[0];
            $prods0[] = $prod[0];
            $prods1[] = $prod[1];
        }
        $actions0 = $la->array($actions0);
        $actions1 = $la->array($actions1);
        $prods0 = $la->array($prods0);
        $prods1 = $la->array($prods1);
        $plt->plot($actions0);
        $plt->plot($actions1);
        $plt->plot($prods0);
        $plt->plot($prods1);
        $plt->plot($la->fill($upper_bound[0],$la->alloc([$steps])));
        $plt->plot($la->fill($mean[0],$la->alloc([$steps])));
        $plt->plot($la->fill($lower_bound[0],$la->alloc([$steps])));
        $plt->legend(['action0','action1','prod0','prod1','upper','mean','lower']);
        $plt->title('Normal');
        $plt->show();
        $this->assertTrue(true);
    }


}