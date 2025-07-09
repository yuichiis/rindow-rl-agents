<?php
namespace RindowTest\RL\Agents\Agent\DQNTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\DQN\DQN;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestPolicy implements Policy
{
    protected NDArray $fixedActions;

    public function __construct($fixedActions)
    {
        $this->fixedActions = $fixedActions;
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    public function register(?EventManager $eventManager=null) : void
    {}

    public function initialize() : void // : Operation
    {}

    public function actions(Estimator $network, NDArray $states, bool $training, ?NDArray $masks) : NDArray
    {
        if($states->ndim()!=2) {
            throw new \Exception("states is supposed to be 2D");
        }
        return $this->fixedActions;
    }
}


class DQNTest extends TestCase
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

    public function testActionSingle()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $fixedActions = [0,1];
        foreach($fixedActions as $policyActionOutput) {
            $policyActionOutput = $la->array([$policyActionOutput],dtype:NDArray::int32);
            $policy = new TestPolicy($policyActionOutput);
            $agent = new DQN($la,policy:$policy,nn:$nn, stateShape:[1], numActions:2,fcLayers:[100]);
            for($i=0;$i<100;$i++) {
                $state = $la->array([1]);
                $a = $agent->action($state,training:true);
                //echo "a=".$mo->toString($a,indent:true)."\n";
                //echo "fixedAction=".$mo->toString($policyActionOutput,indent:true)."\n";
                $this->assertInstanceof(NDArray::class,$a);
                $this->assertEquals([],$a->shape());
                $this->assertEquals(NDArray::int32,$a->dtype());
                $this->assertEquals([1],$policyActionOutput->shape());
                $this->assertEquals(
                    $la->scalar($la->squeeze($policyActionOutput)),
                    $la->scalar($la->squeeze($a))
                );
            }
        }
    }

    public function testActionParallel()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $fixedActions = [[0,1],[1,0]];
        foreach($fixedActions as $policyActionOutput) {
            $policyActionOutput = $la->array($policyActionOutput,dtype:NDArray::int32);
            $policy = new TestPolicy($policyActionOutput);
            $agent = new DQN($la,policy:$policy,nn:$nn, stateShape:[1], numActions:2,fcLayers:[100]);
            for($i=0;$i<100;$i++) {
                $states = [$la->array([1]),$la->array([2])]; // parallel
                $a = $agent->action($states,training:true);
                //echo "a=".$mo->toString($a,indent:true)."\n";
                //echo "fixedAction=".$mo->toString($policyActionOutput,indent:true)."\n";
                $this->assertInstanceof(NDArray::class,$a);
                $this->assertEquals([2],$a->shape());
                $this->assertEquals(NDArray::int32,$a->dtype());
                $this->assertEquals([2],$policyActionOutput->shape());
                $this->assertEquals(
                    $policyActionOutput->toArray(),
                    $a->toArray()
                );
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
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100]);
        $mem = new ReplayBuffer($la,$maxsize=2);
        //[$state,$action,$nextState,$reward,$done,$truncated,$info]
        $mem->add([$la->array([0]),$la->array(1,dtype:NDArray::int32),$la->array([1]),1,false,false,[]]);
        $mem->add([$la->array([1]),$la->array(1,dtype:NDArray::int32),$la->array([2]),1,false,false,[]]);
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

    //public function testMaze()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
    //    $plt = new Plot($this->getPlotConfig(),$mo);
    //    //$rules = $la->array([
    //    //    [NAN,    1,    1,  NAN],
    //    //    [NAN,    1,  NAN,    1],
    //    //]);
    //    //
    //    //  +-+-+-+
    //    //  |0 1 2|
    //    //  + + +-+
    //    //  |3|4 5|
    //    //  + +-+ +
    //    //  |6 7|8|
    //    //  +-+-+-+
    //    $rules = $la->array([
    //    //      UP   DOWN  RIGHT   LEFT
    //        [false,  true,  true, false], // 0
    //        [false,  true,  true,  true], // 1
    //        [false, false, false,  true], // 2
    //        [ true,  true, false, false], // 3
    //        [ true, false,  true, false], // 4
    //        [false,  true, false,  true], // 5
    //        [ true, false,  true, false], // 6
    //        [false, false, false,  true], // 7
    //        [ true, false, false, false], // 8
    //    ], dtype:NDArray::bool);
    //    [$width,$height,$exit] = [3,3,8];
    //    $stateFunc = function($env,$x,$done) use ($la) {
    //        return $la->expandDims($x,axis:-1);
    //    };
    //    $env = new Maze($la,$rules,$width,$height,$exit,$throw=true,$maxEpisodeSteps=100);
    //    $stateShape = [1];
    //    $numActions = $env->actionSpace()->n();
    //    //$policy = new AnnealingEpsGreedy($la,start:$espstart=1.0,stop:$stop=0.01,decayRate:$decayRate=0.01);
    //    $agent = new DQN(
    //        $la,
    //        gamma:0.9,targetUpdatePeriod:5,targetUpdateTau:0.05,ddqn:true,
    //        batchSize:32, epsStart:0.9, epsStop:0.05, epsDecayRate:0.07, 
    //        nn:$nn, stateShape:$stateShape, numActions:$numActions, fcLayers:[100]
    //    );
    //    $driver = new EpisodeRunner($la,$env,$agent,experienceSize:10000);
    //    $driver->setCustomStateFunction($stateFunc);
//
    //    $numIterations=200;
    //    $evalInterval=10;
    //    $driver->agent()->initialize();
    //    $history = $driver->train(
    //        numIterations:$numIterations, metrics:['steps','loss'],
    //        evalInterval:$evalInterval, numEvalEpisodes:10, verbose:0);
    //    $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
    //    $plt->plot($ep,$la->array($history['steps']))[0];
    //    $plt->legend(['steps']);
    //    $plt->title('DQN-maze');
    //    $plt->show();
    //    $this->assertTrue(true);
    //}
}
