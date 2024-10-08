<?php
namespace RindowTest\RL\Agents\Agent\DqnTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\EventManager;
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

    public function register(EventManager $eventManager=null) : void
    {}

    public function initialize() // : Operation
    {}

    public function action(QPolicy $network, NDArray $values,bool $training) : NDArray
    {
        return $this->fixedAction;
    }
}


class DqnTest extends TestCase
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
            'renderer.execBackground' => true,
        ];
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $fixedActions = [0,1];
        foreach($fixedActions as $fixedAction) {
            $fixedAction = $la->array([$fixedAction]);
            $policy = new TestPolicy($fixedAction);
            $agent = new DQN($la,policy:$policy,nn:$nn, obsSize:[1], numActions:2,fcLayers:[100]);
            for($i=0;$i<100;$i++) {
                $a = $agent->action($la->array([1]),$training=true);
                echo $mo->toString($a,indent:true)."\n";
                echo $mo->toString($fixedAction,indent:true)."\n";
                $this->assertEquals($fixedAction,$a);
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
