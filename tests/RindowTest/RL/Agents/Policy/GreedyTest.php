<?php
namespace RindowTest\RL\Agents\Policy\GreedyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Policy\Greedy;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use LogicException;
use InvalidArgumentException;

class TestQPolicy implements QPolicy
{
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
        $values = $la->gather($this->prob,$state,$axis=null);
        return $values;
    }

    public function sample(NDArray $state) : NDArray
    {
        $la = $this->la;
        $count = count($state);
        return $la->fill(1,$la->alloc([$count,1],NDArray::uint32));
    }
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
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $probs = $la->array([
            [1,  0],
            [0,  1],
        ]);
        $qpolicy = new TestQPolicy($la,$probs);
        $policy = new Greedy($la);
        $buf = new ReplayBuffer($la,$maxSize=100);

        $a = [];
        $obs = $la->array([[0]]);
        for($i=0;$i<10;$i++) {
            $actions = $policy->action($qpolicy,$obs,true);
            $this->assertEquals([1,1],$actions->shape());
            $this->assertEquals(NDArray::uint32,$actions->dtype());
            $a[] = $actions[0][0];
        }
        $obs = $la->array([[1]]);
        for($i=0;$i<10;$i++) {
            $actions = $policy->action($qpolicy,$obs,true);
            $this->assertEquals([1,1],$actions->shape());
            $this->assertEquals(NDArray::uint32,$actions->dtype());
            $a[] = $actions[0][0];
        }
        $a = $la->array($a);
        $plt->plot($a);
        $plt->legend(['action']);
        $plt->title('Greedy');
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

        $policy = new Greedy($la);

        $this->assertEquals([[0],[1],[2]],$policy->action($qpolicy,$states,true)->toArray());
    }
}