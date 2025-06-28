<?php
namespace RindowTest\RL\Agents\Policy\BoltzmannTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use LogicException;
use InvalidArgumentException;
use function Rindow\Math\Matrix\R;

class TestEstimator implements Estimator
{
    protected $la;
    protected $prob;

    public function __construct($la,NDArray $prob)
    {
        $this->la = $la;
        $this->prob = $prob;
    }

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
    public function getActionValues(NDArray $state) : NDArray
    {
        $la = $this->la;
        $state = $la->squeeze($state,axis:-1);
        //$values = $la->gather($this->prob,$state,$axis=null);
        //echo "state:".$la->shapeToString($state->shape())."\n";
        $values = $la->gatherb($this->prob,$state);
        //echo "values:";var_dump($values->toArray())."\n";
        return $values;
    }

    //public function probabilities(NDArray $state) : NDArray
    //{
    //    throw new \Exception("ILLEGAL operation", 1);
    //}
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
        $estimator = new TestEstimator($la,$probs);
        $policy = new Boltzmann($la);
        $occur = $mo->zeros([2,4]);

        $times = 1000;
        foreach($probs as $state=>$prob) {
            for($i=0;$i<$times;$i++) {
                $states = $la->array([[$state]],dtype:NDArray::int32);
                $actions = $policy->actions($estimator,$states,training:true,masks:null);
                $this->assertEquals([1],$actions->shape());
                $this->assertEquals(NDArray::int32,$actions->dtype());
                $actnum = $actions[0];
                $la->increment($occur[$state][R($actnum,$actnum+1)],1);
            }
        }
        $la->scal(1/$times,$occur);
        //echo "occur:".$mo->toString($occur,format:'%5.3f',indent:true)."\n";

        //echo "orgprobs:".$mo->toString($probs,format:'%5.3f',indent:true)."\n";
        //echo "sumprobs:".$mo->toString($la->reduceSum($probs,axis:-1),format:'%5.3f',indent:true)."\n";
        $la->multiply($la->reciprocal($la->reduceSum($probs,axis:-1)), $probs, trans:true);
        //echo "probs:".$mo->toString($probs,format:'%5.3f',indent:true)."\n";

        $plt->plot($la->transpose($probs));
        $plt->plot($la->transpose($occur));
        $plt->legend(['prob0.1','prob1.0','occur0.1','occur1.0']);
        $plt->title('Boltzmann');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testWithMask()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $probs = $la->array([
            [0.1,  0.2,  0.3,  0.4],
            [1,  2,  3,  4],
        ]);
        $estimator = new TestEstimator($la,$probs);
        $policy = new Boltzmann($la);
        $occur = $mo->zeros([2,4]);

        $masks = $la->array([
            [true,true,true,false],
        ],dtype:NDArray::bool);
        $times = 1000;
        foreach($probs as $state=>$prob) {
            for($i=0;$i<$times;$i++) {
                $states = $la->array([[$state]],dtype:NDArray::int32);
                $actions = $policy->actions($estimator,$states,training:true,masks:$masks);
                $this->assertEquals([1],$actions->shape());
                $this->assertEquals(NDArray::int32,$actions->dtype());
                $actnum = $actions[0];
                $la->increment($occur[$state][R($actnum,$actnum+1)],1);
            }
        }
        $la->scal(1/$times,$occur);
        //echo "occur:".$mo->toString($occur,format:'%5.3f',indent:true)."\n";

        //echo "orgprobs:".$mo->toString($probs,format:'%5.3f',indent:true)."\n";
        //echo "sumprobs:".$mo->toString($la->reduceSum($probs,axis:-1),format:'%5.3f',indent:true)."\n";
        $la->multiply($la->reciprocal($la->reduceSum($probs,axis:-1)), $probs, trans:true);
        //echo "probs:".$mo->toString($probs,format:'%5.3f',indent:true)."\n";

        $plt->plot($la->transpose($probs));
        $plt->plot($la->transpose($occur));
        $plt->legend(['prob0.1','prob1.0','occur0.1','occur1.0']);
        $plt->title('Boltzmann with Masks');
        $plt->show();
        $this->assertTrue(true);
    }

    //public function testAction()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $this->newLa($mo);
    //    $min = 1e-9;
    //    $max = 1e9;
//
    //    $probs = $la->array([
    //        [1.0, 0.0, 0.0],
    //        [0.0, 1.0, 0.0],
    //        [0.0, 0.0, 1.0],
    //    ]);
    //    //$probs = $la->minimum($la->maximum($probs,$min),$max);
//
    //    $estimator = new TestEstimator($la,$probs);
    //    $policy = new Boltzmann($la,tau:1.0);
//
    //    $states = $la->array([
    //        [2],
    //        [1],
    //        [0]
    //    ],dtype:NDArray::int32);
//
    //    $this->assertEquals([2,1,0],$policy->actions($estimator,$states,training:true,masks:null)->toArray());
//
    //    $this->assertEquals([2,1,0],$policy->actions($estimator,$states,training:false,masks:null)->toArray());
    //}
}