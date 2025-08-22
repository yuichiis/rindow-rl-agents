<?php
namespace RindowTest\RL\Agents\Policy\EpsilonGreedyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Policy\EpsilonGreedy;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use LogicException;
use InvalidArgumentException;

class TestEstimator implements Estimator
{
    public function __construct(
        protected object $la,
        protected NDArray $values,
        protected bool $noRules,
    )
    {}

    public function stateShape() : array
    {
        return [1];
    }

    public function numActions() : int
    {
        return 2;
    }

    /**
    * @param NDArray $state
    * @return NDArray $qValues
    */
    public function getActionValues(NDArray $state,?bool $std=null) : NDArray|array
    {
        $la = $this->la;
        $state = $la->squeeze($state,axis:-1);
        //$values = $la->gather($this->values,$state,$axis=null);
        $values = $la->gatherb($this->values,$state);
        return $values;
    }

    //public function probabilities(NDArray $state) : ?NDArray
    //{
    //    if($this->noRules) {
    //        return null;
    //    }
    //    $la = $this->la;
    //    $count = count($state);
    //    $prob = $la->repeat($la->array([0,1],dtype:NDArray::float32),$count,axis:-1);
    //    return $prob;
    //}
}

class EpsilonGreedyTest extends TestCase
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

    public function testNoRulesNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $epsilon = 0.2;
        $values = $la->array([
            [1,  0],
            [0,  1],
        ]);
        $estimator = new TestEstimator($la,$values,noRules:true);
        $policy = new EpsilonGreedy($la,epsilon:$epsilon);
        $buf = new ReplayBuffer($la,$maxSize=100);

        $avg = [];
        $states = $la->array([[0]],dtype:NDArray::int32);
        for($i=0;$i<1000;$i++) {
            $actions = $policy->actions($estimator,$states,training:true,masks:null);
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $buf->add($actions[0]);
            $avg[] = array_sum($buf->sample($buf->size()))/$buf->size();
        }
        $avg = $la->array($avg);
        $plt->plot($avg);
        $plt->plot($la->fill(1/$estimator->numActions()*$epsilon,$la->alloc([1000])));
        $plt->legend(['action','epsilon']);
        $plt->title('EpsilonGreedy');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testWithRulesNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $epsilon = 0.2;
        $values = $la->array([
            [1,  0],
            [0,  1],
        ]);
        $estimator = new TestEstimator($la,$values,noRules:false);
        $policy = new EpsilonGreedy($la,epsilon:$epsilon);
        $buf = new ReplayBuffer($la,$maxSize=100);

        $avg = [];
        $states = $la->array([[0]],dtype:NDArray::int32);
        $masks = $la->array([[true,true]],dtype:NDArray::bool);
        for($i=0;$i<1000;$i++) {
            $actions = $policy->actions($estimator,$states,training:true,masks:$masks);
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $buf->add($actions[0]);
            $avg[] = array_sum($buf->sample($buf->size()))/$buf->size();
        }
        $avg = $la->array($avg);
        $plt->plot($avg);
        $plt->plot($la->fill(1/$estimator->numActions()*$epsilon,$la->alloc([1000])));
        $plt->legend(['action','epsilon']);
        $plt->title('EpsilonGreedy with Rules');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);

        $values = $la->array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ]);
        $estimator = new TestEstimator($la,$values,noRules:false);
        $states = $la->array([[0],[1],[2]],dtype:NDArray::int32);

        // always selecting maxvalue
        $policy = new EpsilonGreedy($la,epsilon:0.0);
        $this->assertEquals([0,1,2],$policy->actions($estimator,$states,training:true,masks:null)->toArray());

        // always selecting random
        $policy = new EpsilonGreedy($la,epsilon:1.0);
        $actions = $policy->actions($estimator,$states,training:true,masks:null);
        $this->assertEquals([3],$actions->shape());
        $this->assertEquals(NDArray::int32,$actions->dtype());
    }
}