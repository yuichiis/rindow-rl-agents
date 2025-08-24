<?php
namespace RindowTest\RL\Agents\Policy\GreedyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Policy\Greedy;
use Rindow\RL\Agents\ReplayBuffer\QueueBuffer;
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
    //    return $la->fill(0.5,$la->alloc([$count,$this->values->shape()[1]],NDArray::float32));
    //}
}

class GreedyTest extends TestCase
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

        $probs = $la->array([
            [1,  0],
            [0,  1],
        ]);
        $estimator = new TestEstimator($la,$probs,noRules:true);
        $policy = new Greedy($la);
        $buf = new QueueBuffer($la,$maxSize=100);

        $a = [];
        $states = $la->array([[0]],dtype:NDArray::int32);
        for($i=0;$i<10;$i++) {
            $actions = $policy->actions($estimator,$states,training:true,masks:null);
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $a[] = $actions[0];
        }
        $states = $la->array([[1]],dtype:NDArray::int32);
        for($i=0;$i<10;$i++) {
            $actions = $policy->actions($estimator,$states,training:true,masks:null);
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $a[] = $actions[0];
        }
        $a = $la->array($a);
        $plt->plot($a);
        $plt->legend(['action']);
        $plt->title('Greedy');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testWithRulesNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $probs = $la->array([
            [1,  0],
            [0,  1],
        ]);
        $estimator = new TestEstimator($la,$probs,noRules:true);
        $policy = new Greedy($la);
        $buf = new QueueBuffer($la,$maxSize=100);

        $a = [];
        $states = $la->array([[0]],dtype:NDArray::int32);
        $masks = $la->array([[true,true]],dtype:NDArray::bool);
        for($i=0;$i<10;$i++) {
            $actions = $policy->actions($estimator,$states,training:true,masks:$masks);
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $a[] = $actions[0];
        }
        $states = $la->array([[1]],dtype:NDArray::int32);
        $masks = $la->array([[true,true]],dtype:NDArray::bool);
        for($i=0;$i<10;$i++) {
            $actions = $policy->actions($estimator,$states,training:true,masks:$masks);
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $a[] = $actions[0];
        }
        $a = $la->array($a);
        $plt->plot($a);
        $plt->legend(['action']);
        $plt->title('Greedy with Rules');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);

        $probs = $la->array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ]);
        $estimator = new TestEstimator($la,$probs,noRules:false);
        $states = $la->array([[0],[1],[2]],dtype:NDArray::int32);

        $policy = new Greedy($la);

        $this->assertEquals([0,1,2],$policy->actions($estimator,$states,training:true,masks:null)->toArray());
    }
}