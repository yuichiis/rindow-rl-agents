<?php
namespace RindowTest\RL\Agents\Agent\A2CTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\A2C\A2C;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestPolicy implements Policy
{
    public function __construct($fixedAction)
    {
        $this->fixedAction = $fixedAction;
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    public function register(?EventManager $eventManager=null) : void
    {}

    public function initialize() : void // : Operation
    {}

    public function actions(Estimator $network, NDArray $values, bool $training, ?NDArray $masks) : NDArray
    {
        return $this->fixedAction;
    }
}


class A2CTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newLa($mo)
    {
        return $mo->la();
    }

    public function newBuilder($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    public function testActionOnBoltzmann()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $policy = new Boltzmann($la);

        $agent = new A2C($la,
            policy:$policy,
            nn:$nn, stateShape:[1], numActions:2,fcLayers:[100]);
        $states = [
            $la->array([0]),
            $la->array([1]),
        ];
        for($i=0;$i<100;$i++) {
            $actions = $agent->action($states,training:true);
            $this->assertEquals([2],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
        }
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $agent = new A2C($la,
            batchSize:3,
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100]
        );
        $mem = new ReplayBuffer($la,$maxsize=3);
        //[$state,$action,$nextState,$reward,$done,$info]
        $losses = [];
        for($i=0;$i<100;$i++) {
            $mem->add([$la->array([0]),$la->array(1,dtype:NDArray::int32),$la->array([1]),1,false,false,[]]);
            $mem->add([$la->array([1]),$la->array(1,dtype:NDArray::int32),$la->array([2]),1,false,false,[]]);
            $mem->add([$la->array([2]),$la->array(0,dtype:NDArray::int32),$la->array([3]),1,false,false,[]]);
            $losses[] = $agent->update($mem);
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('A2C');
        $plt->show();
        $this->assertTrue(true);
    }
}
