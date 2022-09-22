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
        return 2;
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

        $epsilon = 0.2;
        $qpolicy = new TestQPolicy($la);
        $policy = new EpsilonGreedy($la,$qpolicy,epsilon:$epsilon);
        $buf = new ReplayBuffer($la,$maxSize=100);

        $avg = [];
        for($i=0;$i<1000;$i++) {
            $buf->add($policy->action([0],true));
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

        $qpolicy = new TestQPolicy($la);

        $policy = new EpsilonGreedy($la,$qpolicy,epsilon:0.0);
        $this->assertEquals(0,$policy->action([1,0,0],true));
        $this->assertEquals(1,$policy->action([0,1,0],true));
        $this->assertEquals(2,$policy->action([0,0,1],true));

        $policy = new EpsilonGreedy($la,$qpolicy,epsilon:1.0);
        $this->assertEquals(0,$policy->action([1,0,0],false));
        $this->assertEquals(1,$policy->action([0,1,0],false));
        $this->assertEquals(2,$policy->action([0,0,1],false));
    }
}