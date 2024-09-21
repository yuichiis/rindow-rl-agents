<?php
namespace RindowTest\RL\Agents\Policy\EpsilonGreedyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Policy\EpsilonGreedy;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use LogicException;
use InvalidArgumentException;

class TestQPolicy implements QPolicy
{
    protected $la;
    protected $prob;

    public function __construct($la,NDArray $prob)
    {
        $this->la = $la;
        $this->prob = $prob;
    }

    public function obsSize()
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
    public function getQValues(NDArray $state) : NDArray
    {
        $la = $this->la;
        $state = $la->squeeze($state,$axis=-1);
        //$values = $la->gather($this->prob,$state,$axis=null);
        $values = $la->gatherb($this->prob,$state);
        return $values;
    }

    public function sample(NDArray $state) : NDArray
    {
        $la = $this->la;
        $count = count($state);
        return $la->fill(1,$la->alloc([$count,1],NDArray::uint32));
    }
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
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $epsilon = 0.2;
        $probs = $la->array([
            [1,  0],
            [0,  1],
        ]);
        $qpolicy = new TestQPolicy($la,$probs);
        $policy = new EpsilonGreedy($la,epsilon:$epsilon);
        $buf = new ReplayBuffer($la,$maxSize=100);

        $avg = [];
        $obs = $la->array([[0]]);
        for($i=0;$i<1000;$i++) {
            $actions = $policy->action($qpolicy,$obs,true);
            $this->assertEquals([1,1],$actions->shape());
            $this->assertEquals(NDArray::uint32,$actions->dtype());
            $buf->add($actions[0][0]);
            $avg[] = array_sum($buf->sample($buf->size()))/$buf->size();
        }
        $avg = $la->array($avg);
        $plt->plot($avg);
        $plt->plot($la->fill(1/$qpolicy->numActions()*$epsilon,$la->alloc([1000])));
        $plt->legend(['action','epsilon']);
        $plt->title('EpsilonGreedy');
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
        $qpolicy = new TestQPolicy($la,$probs);
        $states = $la->array([[0],[1],[2]]);

        $policy = new EpsilonGreedy($la,epsilon:0.0);
        $this->assertEquals([[0],[1],[2]],$policy->action($qpolicy,$states,true)->toArray());

        $policy = new EpsilonGreedy($la,epsilon:1.0);
        $this->assertEquals([[0],[1],[2]],$policy->action($qpolicy,$states,false)->toArray());
    }
}