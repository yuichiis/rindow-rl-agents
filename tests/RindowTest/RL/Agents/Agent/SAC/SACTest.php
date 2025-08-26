<?php
namespace RindowTest\RL\Agents\Agent\DDPG\SACTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\ReplayBuffer\QueueBuffer;
use Rindow\RL\Agents\Agent\SAC\SAC;
use Rindow\RL\Agents\Agent\SAC\ActorNetwork;
use Rindow\RL\Agents\Agent\SAC\CriticNetwork;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestPolicy implements Policy
{
    protected NDArray $fixedAction;

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


class SACTest extends TestCase
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

    public function testBuildNetwork()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        $stateShape=[3];
        $numActions=1;
        $lower_bound=$la->array([-2]);
        $upper_bound=$la->array([2]);
        $critic_lr=0.002;
        $actor_lr=0.001;
        //$actorNetworkOptions = [
        //    'fcLayers' => [256, 256],
        //    'outputActivation' => 'tanh',
        //    'minval' => -0.003,
        //    'maxval' => 0.003,
        //];
        //$criticNetworkOptions = [
        //    'staFcLayers' => [128],
        //    'actLayers' => [128],
        //];
        $agent = new SAC($la,$nn,
            $stateShape,
            numActions:$numActions,
            lowerBound:$lower_bound,
            upperBound:$upper_bound,
            batchSize:2,
            gamma:0.99,
            targetUpdatePeriod:1,
            targetUpdateTau:0.005,
            //actorNetworkOptions:$actorNetworkOptions,
            //criticNetworkOptions:$criticNetworkOptions,
            criticOptimizerOpts:['lr'=>$critic_lr],
            actorOptimizerOpts:['lr'=>$actor_lr],
        );
        $this->assertInstanceof(ActorNetwork::class,$agent->actorNetwork());
        $this->assertInstanceof(ActorNetwork::class,$agent->targetActorNetwork());
        $this->assertInstanceof(CriticNetwork::class,$agent->criticNetwork());
        $this->assertInstanceof(CriticNetwork::class,$agent->targetcriticNetwork());
        //$agent->summary();
        $this->assertTrue(true);


        // defaults
        $stateShape=[3];
        $numActions=1;
        $lower_bound=$la->array([-2]);
        $upper_bound=$la->array([2]);

        $agent = new SAC($la,$nn,
            $stateShape,$numActions,$lower_bound,$upper_bound,
        );
        //$agent->summary();
        $this->assertTrue(true);
    }

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        //
        $stateShape=[1];
        $numActions=2;
        $lower_bound=$la->array([-2,-3]);
        $upper_bound=$la->array([2,3]);
        $fixedActions = [-1,1];
        foreach($fixedActions as $fixedAction) {
            $agent = new SAC(
                $la,$nn,
                $stateShape,
                $numActions,
                $lower_bound,
                $upper_bound,
            );
            for($i=0;$i<100;$i++) {
                $state = $la->array([1]);
                $action = $agent->action($state,training:true);
                $this->assertInstanceof(NDArray::class,$action);
                $this->assertEquals([$numActions],$action->shape());
                $this->assertEquals(NDArray::float32,$action->dtype());
            }
        }
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        //
        $stateShape=[3];
        $numActions=1;
        $lower_bound=$la->array([-2]);
        $upper_bound=$la->array([2]);
        $agent = new SAC(
            $la,$nn,
            $stateShape,$numActions,$lower_bound,$upper_bound,
            batchSize:2,
            gamma:0.99,
            autoTuneAlpha:true,
            targetUpdatePeriod:1,
            targetUpdateTau:0.005,
        );
        $mem = new QueueBuffer($la,$maxsize=2);
        //[$state,$action,$nextState,$reward,$done,$info]
        $mem->add([$la->array([0,0,0]),$la->array([0.1]),$la->array([0,0.1,0.1]),-0.1,false,false,[]]);
        $mem->add([$la->array([0,0,0]),$la->array([0.1]),$la->array([0,0.1,0.1]),-0.1,false,false,[]]);
        //$mem->add([$la->array([0,0.1,0.1]),$la->array([0.2]),$la->array([0,0.2,0.1]),-0.1,false,[]]);
        //$mem->add([$la->array([0,0.2,0.1]),$la->array([0.1]),$la->array([0,0.3,0.3]),-0.1,false,[]]);
        $losses = [];
        for($i=0;$i<100;$i++) {
            $losses[] = $agent->update($mem);
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('DDPG');
        $plt->show();
        $this->assertTrue(true);
    }
}
