<?php
namespace RindowTest\RL\Agents\Agent\PPO\PPOTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\PPO\PPO;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\RL\Agents\Util\Metrics;
use Rindow\RL\Gym\Core\Spaces\Box;
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


class PPOTest extends TestCase
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

        $agent = new PPO($la,
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

        $agent = new PPO($la,
            batchSize:3,epochs:4,rolloutSteps:3,
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100],
            normAdv:true,
        );
        $metrics = new Metrics();
        $agent->setMetrics($metrics);
        $metrics->attract(['loss','entropy']);
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
        $plt->title('PPO');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testActionContinuous()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $actionSpace = new Box($la,high:1,low:-1,shape:[2]);
        $agent = new PPO($la,
            continuous:true,
            nn:$nn, stateShape:[1], actionSpace:$actionSpace, fcLayers:[100],
        );
        $states = [
            $la->array([0]),
            $la->array([1]),
        ];
        for($i=0;$i<100;$i++) {
            $actions = $agent->action($states,training:true);
            $this->assertEquals([2,2],$actions->shape());
            $this->assertEquals(NDArray::float32,$actions->dtype());
        }
    }

    public function testUpdateContinuous()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $actionSpace = new Box($la,high:1,low:-1,shape:[2]);
        $agent = new PPO($la,
            batchSize:3,epochs:4,rolloutSteps:3,
            nn:$nn, stateShape:[1], actionSpace:$actionSpace, fcLayers:[100],
            normAdv:true,
            continuous:true,
        );
        $metrics = new Metrics();
        $agent->setMetrics($metrics);
        $metrics->attract(['loss','entropy']);
        $mem = new ReplayBuffer($la,$maxsize=3);
        //[$state,$action,$nextState,$reward,$done,$info]
        $losses = [];
        for($i=0;$i<100;$i++) {
            $mem->add([$la->array([0]),$la->array([1,1]),$la->array([1]),1,false,false,[]]);
            $mem->add([$la->array([1]),$la->array([1,1]),$la->array([2]),1,false,false,[]]);
            $mem->add([$la->array([2]),$la->array([0,0]),$la->array([3]),1,false,false,[]]);
            $losses[] = $agent->update($mem);
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('PPO');
        $plt->show();
        $this->assertTrue(true);
    }
}
