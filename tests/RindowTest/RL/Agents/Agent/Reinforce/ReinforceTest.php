<?php
namespace RindowTest\RL\Agents\Agent\Reinforce\ReinforceTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\Reinforce\Reinforce;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestPolicy implements Policy
{
    protected $fixedAction;

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

    public function actions(Estimator $network, NDArray $values,bool $training, ?NDArray $masks) : NDArray
    {
        return $this->fixedAction;
    }
}


class ReinforceTest extends TestCase
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
            $policy = new TestPolicy($la->array([$fixedAction],dtype:NDArray::int32));
            $agent = new Reinforce($la,policy:$policy,nn:$nn, stateShape:[1], numActions:2,fcLayers:[100]);
            for($i=0;$i<100;$i++) {
                $action = $agent->action($la->array([1]),training:true);
                $actionNumber = $la->scalar($la->squeeze($action));
                $this->assertEquals($fixedAction,$actionNumber);
            }
        }
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $agent = new Reinforce($la,
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100],
            mo:$mo,
        );
        $mem = new ReplayBuffer($la,$maxsize=10000);
        $losses = [];
        for($i=0;$i<5;$i++) {
            //[$state,$action,$nextState,$reward,$done,$discontinued,$info]
            $mem->add([$la->array([1]),$la->array(1,dtype:NDArray::int32),$la->array([1]),1,false,false,[]]);
            $mem->add([$la->array([1]),$la->array(1,dtype:NDArray::int32),$la->array([2]),1,false,false,[]]);
            $losses[] = $agent->update($mem);
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('Reinforce');
        $plt->show();
        $this->assertTrue(true);
    }
}
