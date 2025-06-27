<?php
namespace RindowTest\RL\Agents\Agent\PolicyGradientTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PolicyGradient\PolicyGradient;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class PolicyGradientTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newLa($mo)
    {
        return $mo->la();
    }

    public function newRules($la)
    {
        return $rules;
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    //public function testAction()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
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
    //    $agent = new PolicyGradient($la,$rules,$eta=0.1,mo:$mo);
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
        $rules = $la->array([
            [true,  false, false, false],
            [false, true,  true,  false],
            [false, false, false, true ],
            [true,  false, false, true ],
        ],dtype:NDArray::bool);
        [$numStates,$numActions] = $rules->shape();
        $fixedActions = [
            [0],
            [1,2],
            [3],
            [0,3],
        ];
        $agent = new PolicyGradient($la,$numStates,$numActions,eta:0.1,mo:$mo);
        foreach($fixedActions as $idx => $actions) {
            for($i=0;$i<100;$i++) {
                $state = $la->array([$idx],dtype:NDArray::int32);
                $info = ['validActions'=>$rules[$idx]];
                $action = $agent->action($state,training:true,info:$info);
                $action = $la->scalar($action);
                //var_dump($action);
                //var_dump($actions);
                $this->assertTrue(in_array($action,$actions));
            }
        }
    }

    //public function testUpdate()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
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
    //    $agent = new PolicyGradient($la,$rules,$eta=0.1,mo:$mo);
    //    $driver = new EpisodeDriver($la,$env,$agent,experienceSize:10000);
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
    //    $plt->title('PolicyGradient');
    //    $plt->show();
    //    $this->assertTrue(true);
    //}

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $plt = new Plot($this->getPlotConfig(),$mo);

        //  +-+-+-+
        //  |0 1 2|
        //  + + +-+
        //  |3|4 5|
        //  + +-+ +
        //  |6 7|8|
        //  +-+-+-+
        $rules = $la->array([
        //   UP    DOWN  RIGHT LEFT
            [false,  true,  true, false], // 0
            [false,  true,  true,  true], // 1
            [false, false, false,  true], // 2
            [ true,  true, false, false], // 3
            [ true, false,  true, false], // 4
            [false,  true, false,  true], // 5
            [ true, false,  true, false], // 6
            [false, false, false,  true], // 7
            [ true, false, false, false], // 8
        ], dtype:NDArray::bool);
        [$numStates,$numActions] = $rules->shape();
        [$width,$height,$exit] = [3,3,8];
        $stateFunc = function($env,$x,$done) use ($la) {
            return $la->expandDims($x,axis:-1);
        };
        $env = new Maze($la,$rules,$width,$height,$exit,$throw=true,$maxEpisodeSteps=100);
        $agent = new PolicyGradient($la,$numStates,$numActions,$eta=0.1,mo:$mo);
        $driver = new EpisodeDriver($la,$env,$agent,experienceSize:10000);
        $driver->setCustomStateFunction($stateFunc);

        $numIterations=200;
        $evalInterval=10;
        $driver->agent()->initialize();
        $history = $driver->train(
            numIterations:$numIterations, metrics:['steps','loss'],
            evalInterval:$evalInterval, numEvalEpisodes:10, verbose:0);
        $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
        $plt->plot($ep,$la->array($history['steps']))[0];
        $plt->legend(['steps']);
        $plt->title('PolicyGradient');
        $plt->show();
        $this->assertTrue(true);
    }
}
