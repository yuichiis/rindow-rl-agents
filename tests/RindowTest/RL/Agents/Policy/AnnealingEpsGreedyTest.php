<?php
namespace RindowTest\RL\Agents\Policy\AnnealingEpsGreedyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
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
        $policy = new AnnealingEpsGreedy($la,decayRate:0.005);
        $buf = new ReplayBuffer($la,$maxSize=100);

        $epsilon = [];
        $avg = [];
        for($i=0;$i<1000;$i++) {
            $epsilon[] = $policy->getEpsilon();
            $obs = $la->array([[0]]);
            $actions = $policy->action($qpolicy,$obs,true);
            $this->assertEquals([1,1],$actions->shape());
            $this->assertEquals(NDArray::uint32,$actions->dtype());
            $buf->add($actions[0][0]);
            $avg[] = array_sum($buf->sample($buf->size()))/$buf->size();
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
        $qpolicy = new TestQPolicy($la,$probs);
        $states = $la->array([[0],[1],[2]]);

        $policy = new AnnealingEpsGreedy($la,start:0,stop:0);
        $this->assertEquals([[0],[0],[0]],$policy->action($qpolicy,$states,true)->toArray());

        $policy = new AnnealingEpsGreedy($la,start:1,stop:1);
        $this->assertEquals([[1],[1],[1]],$policy->action($qpolicy,$states,true)->toArray());
    }
}