<?php
namespace RindowTest\RL\Agents\Agent\Sarsa\SarsaTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\Sarsa\Sarsa;
use Rindow\RL\Agents\Agent\QLearning\QTable as ValueTable;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\ReplayBuffer\QueueBuffer;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class SarsaTest extends TestCase
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

    public function newRules($la)
    {
        return $rules;
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    //public function testAction()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
    //    $rules = $la->array([
    //        [  1,  NAN,  NAN,  NAN],
    //        [NAN,    1,    1,  NAN],
    //        [NAN,  NAN,  NAN,    1],
    //        [  1,  NAN,  NAN,    1],
    //    ]);
    //    $fixedActions = [
    //        [0],
    //        [1,2],
    //        [3],
    //        [0,3],
    //    ];
    //    $policy = new AnnealingEpsGreedy($la,start:$espstart=1.0,stop:$stop=0.01,decayRate:$decayRate=0.01);
    //    $agent = new Sarsa($la,$rules,$policy,$eta=0.1,$gamma=0.9,mo:$mo);
    //    foreach($fixedActions as $idx => $actions) {
    //        for($i=0;$i<100;$i++) {
    //            $state = $la->array([$idx],dtype:NDArray::int32);
    //            $action = $agent->action($state,training:true);
    //            $action = $la->scalar($action);
    //            $this->assertTrue(in_array($action,$actions));
    //        }
    //    }
    //}

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $rules = $la->array([
            [ true, false, false, false],
            [false,  true,  true, false],
            [false, false, false,  true],
            [ true, false, false,  true],
        ], dtype:NDArray::bool);
        $numStates = $rules->shape()[0];
        $numActions = $rules->shape()[1];
        $fixedActions = [
            [0],
            [1,2],
            [3],
            [0,3],
        ];
        $policy = new AnnealingEpsGreedy($la,start:1.0,stop:0.01,decayRate:0.01);
        $agent = new Sarsa($la,$numStates,$numActions,$policy,eta:0.1,gamma:0.9,stateField:'location',mo:$mo);
        foreach($fixedActions as $idx => $actions) {
            for($i=0;$i<100;$i++) {
                $state = $la->array([$idx],dtype:NDArray::int32);
                $mask = $rules[$idx];
                $obs = ['location'=>$state,'actionMask'=>$mask];
                $action = $agent->action($obs,training:true,info:[]);
                $action = $la->scalar($action);
                $this->assertTrue(in_array($action,$actions));
            }
        }
    }

    //public function testUpdate()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
    //    $plt = new Plot($this->getPlotConfig(),$mo);
    //    $rules = $la->array([
    //        [NAN,    1,    1,  NAN],
    //        [NAN,    1,  NAN,    1],
    //    ]);
    //    $rules = $la->array([
    //    //   UP    DOWN  RIGHT LEFT
    //        [NAN,    1,    1,  NAN], // 0  +-+-+-+
    //        [NAN,    1,    1,    1], // 1  |0 1 2|
    //        [NAN,  NAN,  NAN,    1], // 2  + + +-+
    //        [  1,    1,  NAN,  NAN], // 3  |3|4 5|
    //        [  1,  NAN,    1,  NAN], // 4  + +-+ +
    //        [NAN,    1,  NAN,    1], // 5  |6 7|8|
    //        [  1,  NAN,    1,  NAN], // 6  +-+-+-+
    //        [NAN,  NAN,  NAN,    1], // 7
    //        [  1,  NAN,  NAN,  NAN], // 8
    //    ]);
    //    [$width,$height,$exit] = [3,3,8];
    //    $stateFunc = function($env,$x,$done) use ($la) {
    //        return $la->expandDims($x,axis:-1);
    //    };
    //    $env = new Maze($la,$rules,$width,$height,$exit,$throw=true,$maxEpisodeSteps=100);
    //    $policy = new AnnealingEpsGreedy($la,start:$espstart=1.0,stop:$stop=0.01,decayRate:$decayRate=0.01);
    //    $agent = new Sarsa($la,$rules,$policy,$eta=0.1,$gamma=0.9,mo:$mo);
    //    $driver = new EpisodeRunner($la,$env,$agent,$experienceSize=10000);
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
    //    $plt->title('Sarsa');
    //    $plt->show();
    //    $this->assertTrue(true);
    //}

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        //
        // +-+-+-+
        // |0 1 2|
        // + + +-+
        // |3|4 5|
        // + +-+ +
        // |6 7|8|
        // +-+-+-+
        //
        $rules = $la->array([
        //   UP    DOWN  RIGHT LEFT
            [false,  true,  true, false], // 0
            [false,  true,  true,  true], // 1
            [false, false, false,  true], // 2
            [true,   true, false, false], // 3
            [true,  false,  true, false], // 4
            [false,  true, false,  true], // 5
            [true,  false,  true, false], // 6
            [false, false, false,  true], // 7
            [true,  false, false, false], // 8
        ], dtype:NDArray::bool);
        [$width,$height,$exit] = [3,3,8];
        $stateFunc = function($env,$obs,$done) use ($la,$width) {
            $location = $obs['location'];
            $y = $location[0];
            $x = $location[1];
            $pos = $y*$width + $x;
            $pos = $la->array([$pos],dtype:NDArray::int32);
            $mask = $obs['actionMask'];
            return ['location'=>$pos,'actionMask'=>$mask];
        };
        $env = new Maze($la,$rules,$width,$height,$exit,$throw=true,$maxEpisodeSteps=100);
        $evalEnv = new Maze($la,$rules,$width,$height,$exit,$throw=true,$maxEpisodeSteps=100);
        $numStates = $width*$height;
        $numActions = $env->actionSpace()->n();
        $policy = new AnnealingEpsGreedy($la,start:$espstart=1.0,stop:$stop=0.01,decayRate:$decayRate=0.01);
        $agent = new Sarsa($la,$numStates,$numActions,policy:$policy,eta:0.1,gamma:0.9,stateField:'location',mo:$mo);
        $agent->setCustomStateFunction($stateFunc);
        $driver = new EpisodeRunner($la,$env,$agent,experienceSize:10000,evalEnv:$evalEnv);

        $numIterations=200;
        $evalInterval=10;
        $driver->agent()->initialize();
        $history = $driver->train(
            numIterations:$numIterations, metrics:['steps','loss'],
            evalInterval:$evalInterval, numEvalEpisodes:10, verbose:0);
        $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
        $plt->plot($ep,$la->array($history['steps']))[0];
        $plt->legend(['steps']);
        $plt->title('Sarsa');
        $plt->show();
        $this->assertTrue(true);
    }
}
