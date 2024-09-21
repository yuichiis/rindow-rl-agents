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
use function Rindow\Math\Matrix\R;

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
        throw new \Exception("ILLEGAL operation", 1);
    }
}

class BoltzmannTest extends TestCase
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
            [0.1,  0.2,  0.3,  0.4],
            [1,  2,  3,  4],
        ]);
        $qpolicy = new TestQPolicy($la,$probs);
        $policy = new Boltzmann($la);
        $occur = $mo->zeros([2,4]);

        $times = 1000;
        foreach($probs as $state=>$prob) {
            for($i=0;$i<$times;$i++) {
                $obs = $la->array([[$state]]);
                $actions = $policy->action($qpolicy,$obs,true);
                $this->assertEquals([1,1],$actions->shape());
                $this->assertEquals(NDArray::uint32,$actions->dtype());
                $actnum = $actions[0][0];
                $la->increment($occur[$state][R($actnum,$actnum+1)],1);
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
        $policy = new Boltzmann($la,tau:1.0);

        $values = $la->array([[2],[1],[0]]);

        $this->assertEquals([[2],[1],[0]],$policy->action($qpolicy,$values,true)->toArray());

        $this->assertEquals([[2],[1],[0]],$policy->action($qpolicy,$values,false)->toArray());
    }
}