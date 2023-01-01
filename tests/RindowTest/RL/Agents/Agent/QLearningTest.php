<?php
namespace RindowTest\RL\Agents\Agent\QLearningTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\QLearning;
use Rindow\RL\Agents\Network\QTable;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

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

    public function newRules($la)
    {
        return $rules;
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
        $rules = $la->array([
            [  1,  NAN,  NAN,  NAN],
            [NAN,    1,    1,  NAN],
            [NAN,  NAN,  NAN,    1],
            [  1,  NAN,  NAN,    1],
        ]);
        $fixedActions = [
            [0],
            [1,2],
            [3],
            [0,3],
        ];
        $qtable = new QTable($la,$rules);
        $policy = new AnnealingEpsGreedy($la,start:$espstart=1.0,stop:$stop=0.01,decayRate:$decayRate=0.01);
        $agent = new QLearning($la,$qtable,$policy,$eta=0.1,$gamma=0.9,mo:$mo);
        foreach($fixedActions as $obs => $actions) {
            for($i=0;$i<100;$i++) {
                $action = $agent->action($obs,$training=true);
                $this->assertTrue(in_array($action,$actions));
            }
        }
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $rules = $la->array([
            [NAN,    1,    1,  NAN],
            [NAN,    1,  NAN,    1],
        ]);
        $rules = $la->array([
        //   UP    DOWN  RIGHT LEFT
            [NAN,    1,    1,  NAN], // 0  +-+-+-+
            [NAN,    1,    1,    1], // 1  |0 1 2|
            [NAN,  NAN,  NAN,    1], // 2  + + +-+
            [  1,    1,  NAN,  NAN], // 3  |3|4 5|
            [  1,  NAN,    1,  NAN], // 4  + +-+ +
            [NAN,    1,  NAN,    1], // 5  |6 7|8|
            [  1,  NAN,    1,  NAN], // 6  +-+-+-+
            [NAN,  NAN,  NAN,    1], // 7
            [  1,  NAN,  NAN,  NAN], // 8
        ]);
        [$width,$height,$exit] = [3,3,8];
        $env = new Maze($la,$rules,$width,$height,$exit,$throw=true,$maxEpisodeSteps=100);
        $qtable = new QTable($la,$rules);
        $policy = new AnnealingEpsGreedy($la,start:$espstart=1.0,stop:$stop=0.01,decayRate:$decayRate=0.01);
        $agent = new QLearning($la,$qtable,$policy,$eta=0.1,$gamma=0.9,mo:$mo);
        $driver = new EpisodeDriver($la,$env,$agent,$experienceSize=10000);

        $numIterations=200;
        $evalInterval=10;
        $driver->agent()->initialize();
        $history = $driver->train(
            numIterations:$numIterations, metrics:['steps','loss'],
            evalInterval:$evalInterval, numEvalEpisodes:10, verbose:0);
        $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
        $plt->plot($ep,$la->array($history['steps']))[0];
        $plt->legend(['steps']);
        $plt->title('QLearning');
        $plt->show();
        $this->assertTrue(true);
    }
}
