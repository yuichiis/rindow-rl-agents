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

        $qpolicy = new TestQPolicy($la);
        $policy = new Greedy($la,$qpolicy);
        $buf = new ReplayBuffer($la,$maxSize=100);

        $avg = [];
        for($i=0;$i<10;$i++) {
            $buf->add($policy->action([0],true));
            $avg[] = array_sum($buf->sample($buf->size()))/$buf->size();
        }
        $avg = $la->array($avg);
        $plt->plot($avg);
        $plt->legend(['action']);
        $plt->title('Greedy');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);

        $qpolicy = new TestQPolicy($la);
        $policy = new Greedy($la,$qpolicy);

        $this->assertEquals(0,$policy->action([1,0,0],true));
        $this->assertEquals(1,$policy->action([0,1,0],true));
        $this->assertEquals(2,$policy->action([0,0,1],true));
    }
}