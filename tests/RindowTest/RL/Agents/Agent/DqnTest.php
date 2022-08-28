<?php
namespace RindowTest\RL\Agents\Agent\DqnTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Agent\DQN;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
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

    public function initialize() // : Operation
    {}

    public function action($values,bool $training, int $time=null)
    {
        return $this->fixedAction;
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

    public function newBuilder($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $fixedActions = [0,1];
        foreach($fixedActions as $fixedAction) {
            $policy = new TestPolicy($fixedAction);
            $agent = new DQN($la,policy:$policy,nn:$nn, obsSize:[1], numActions:2,fcLayers:[100]);
            for($i=0;$i<100;$i++) {
                $this->assertEquals($fixedAction,$agent->action($la->array([1]),$training=true));
            }
        }
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $agent = new DQN($la,
            batchSize:2, epsStart:0, epsStop:0,
            nn:$nn, obsSize:[1], numActions:2, fcLayers:[100]);
        $mem = new ReplayBuffer($la,$maxsize=2);
        //[$observation,$action,$nextObs,$reward,$done,$info]
        $mem->add([0,1,1,1,false,[]]);
        $mem->add([1,1,2,1,false,[]]);
        $losses = [];
        for($i=0;$i<100;$i++) {
            $losses[] = $agent->update($mem);
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('DQN');
        $plt->show();
        $this->assertTrue(true);
    }
}
