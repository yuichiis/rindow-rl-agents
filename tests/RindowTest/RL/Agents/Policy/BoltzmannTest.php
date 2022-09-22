<?php
namespace RindowTest\RL\Agents\Policy\BoltzmannTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Policy\Boltzmann;
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
    public function getQValues($state) : NDArray
    {
        $state = $state[0];
        return $this->prob[$state];
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

        $probs = $la->array([
            [0.1,  0.2,  0.3,  0.4],
            [1,  2,  3,  4],
        ]);
        $qpolicy = new TestQPolicy($la,$probs);
        $policy = new Boltzmann($la,$qpolicy);
        $occur = $mo->zeros([2,4]);

        $times = 1000;
        foreach($probs as $state=>$prob) {
            for($i=0;$i<$times;$i++) {
                $action = $policy->action([$state],true);
                $la->increment($occur[$state][[$action,$action]],1);
            }
        }
        $la->scal(1/$times,$occur);
        $la->multiply($la->reciprocal($la->reduceSum($probs,$axis=1)), $probs, $trans=true);

        $plt->plot($la->transpose($probs));
        $plt->plot($la->transpose($occur));
        $plt->legend(['prob0.1','prob1.0','occur0.1','occur1.0']);
        $plt->title('Boltzmann');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);

        $probs = $la->array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        $qpolicy = new TestQPolicy($la,$probs);
        $policy = new Boltzmann($la,$qpolicy,tau:1.0);

        $this->assertEquals(0,$policy->action([0],true));
        $this->assertEquals(1,$policy->action([1],true));
        $this->assertEquals(2,$policy->action([2],true));

        $this->assertEquals(0,$policy->action([0],false));
        $this->assertEquals(1,$policy->action([1],false));
        $this->assertEquals(2,$policy->action([2],false));
    }
}