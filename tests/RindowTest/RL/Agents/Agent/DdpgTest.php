<?php
namespace RindowTest\RL\Agents\Agent\DdpgTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Agent\Ddpg;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Network\ActorNetwork;
use Rindow\RL\Agents\Network\CriticNetwork;
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

    public function testActorNetwork()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        $obsSize=[3];
        $actionSize=[1];
        $lower_bound=$la->array([-2]);
        $upper_bound=$la->array([2]);
        $agent = new Ddpg($la,$nn,
            $obsSize,$actionSize,$lower_bound,$upper_bound,
            std_dev:0.2,
            batchSize:2,
            gamma:0.99,
            targetUpdatePeriod:1,
            targetUpdateTau:0.005,
            criticOptimizerOpts:['lr'=>$critic_lr=0.002],
            actorOptimizerOpts:['lr'=>$actor_lr=0.001],
        );
        $this->assertInstanceof(ActorNetwork::class,$agent->actorNetwork());
        $this->assertInstanceof(ActorNetwork::class,$agent->targetActorNetwork());
        $this->assertInstanceof(CriticNetwork::class,$agent->criticNetwork());
        $this->assertInstanceof(CriticNetwork::class,$agent->targetcriticNetwork());
        //$actor->summary();
        $this->assertTrue(true);
    }

    //public function testAction()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
    //    //
    //    $fixedActions = [0,1];
    //    foreach($fixedActions as $fixedAction) {
    //        $policy = new TestPolicy($fixedAction);
    //        $agent = new Ddpg($la,policy:$policy,nn:$nn, obsSize:[1], numActions:2,fcLayers:[100]);
    //        for($i=0;$i<100;$i++) {
    //            $this->assertEquals($fixedAction,$agent->action($la->array([1]),$training=true));
    //        }
    //    }
    //}

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        //
        $obsSize=[3];
        $actionSize=[1];
        $lower_bound=$la->array([-2]);
        $upper_bound=$la->array([2]);
        $agent = new Ddpg($la,$nn,
            $obsSize,$actionSize,$lower_bound,$upper_bound,
            std_dev:0.2,
            batchSize:2,
            gamma:0.99,
            targetUpdatePeriod:1,
            targetUpdateTau:0.005,
            criticOptimizerOpts:['lr'=>$critic_lr=0.002],
            actorOptimizerOpts:['lr'=>$actor_lr=0.001],
        );
        $mem = new ReplayBuffer($la,$maxsize=2);
        //[$observation,$action,$nextObs,$reward,$done,$info]
        $mem->add([$la->array([0,0,0]),$la->array([0.1]),$la->array([0,0.1,0.1]),-0.1,false,[]]);
        $mem->add([$la->array([0,0,0]),$la->array([0.1]),$la->array([0,0.1,0.1]),-0.1,false,[]]);
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
