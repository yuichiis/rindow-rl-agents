<?php
namespace RindowTest\RL\Agents\Policy\AnnealingEpsGreedyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
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
    //    $prob = $la->repeat($la->array([0,1],dtype:NDArray::float32),$count,axis:-1);
    //    return $prob;
    //}
}

class AnnealingEpsGreedyTest extends TestCase
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

        $values = $la->array([
            [1,  0],
            [0,  1],
        ],dtype:NDArray::float32);
        $estimator = new TestEstimator($la,$values,noRules:true);
        $policy = new AnnealingEpsGreedy($la,decayRate:0.005);
        $buf = new QueueBuffer($la,$maxSize=100);

        $epsilon = [];
        $avg = [];
        // states(batch,state) = (1,1)
        $states = $la->array([[0]],dtype:NDArray::int32);
        $this->assertEquals([1,1],$states->shape());
        $this->assertEquals(NDArray::int32,$states->dtype());
        for($i=0;$i<1000;$i++) {
            $epsilon[] = $policy->getEpsilon();
            $actions = $policy->actions($estimator,$states,training:true,masks:null);
            // actions(batchs) = (1)
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $buf->add([$actions[0]]);
            $avg[] = array_sum($buf->sample($buf->size())[0])/$buf->size();
        }
        $epsilon = $la->array($epsilon);
        $avg = $la->array($avg);
        $plt->plot($epsilon);
        $plt->plot($avg);
        $plt->legend(['epsilon','action']);
        $plt->title('AnnealingEpsGreedy');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testWithRulesNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $values = $la->array([
            [1,  0],
            [0,  1],
        ],dtype:NDArray::float32);
        $estimator = new TestEstimator($la,$values,noRules:false);
        $policy = new AnnealingEpsGreedy($la,decayRate:0.005);
        $buf = new QueueBuffer($la,$maxSize=100);

        $epsilon = [];
        $avg = [];
        // states(batch,state) = (1,1)
        $states = $la->array([[0]],dtype:NDArray::int32);
        $masks = $la->array([[true,true]],dtype:NDArray::bool);
        $this->assertEquals([1,1],$states->shape());
        $this->assertEquals([1,2],$masks->shape());
        $this->assertEquals(NDArray::int32,$states->dtype());
        for($i=0;$i<1000;$i++) {
            $epsilon[] = $policy->getEpsilon();
            $actions = $policy->actions($estimator,$states,training:true,masks:$masks);
            // actions(batchs) = (1)
            $this->assertEquals([1],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
            $buf->add([$actions[0]]);
            $avg[] = array_sum($buf->sample($buf->size())[0])/$buf->size();
        }
        $epsilon = $la->array($epsilon);
        $avg = $la->array($avg);
        $plt->plot($epsilon);
        $plt->plot($avg);
        $plt->legend(['epsilon','action']);
        $plt->title('AnnealingEpsGreedy');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);

        $probs = $la->array([
            [1,0],
            [1,0],
            [1,0],
        ]);
        $estimator = new TestEstimator($la,$probs,noRules:false);
        $states = $la->array([[0],[1],[2]],dtype:NDArray::int32);

        // always selecting maxvalue
        $policy = new AnnealingEpsGreedy($la,start:0,stop:0);
        $this->assertEquals([0,0,0],$policy->actions($estimator,$states,training:true,masks:null)->toArray());

        // always selecting random
        $policy = new AnnealingEpsGreedy($la,start:1,stop:1);
        $actions = $policy->actions($estimator,$states,training:true,masks:null);
        $this->assertEquals([3],$actions->shape());
        $this->assertEquals(NDArray::int32,$actions->dtype());

    }
}