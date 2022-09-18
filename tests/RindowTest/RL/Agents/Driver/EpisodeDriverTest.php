<?php
namespace RindowTest\RL\Agents\Driver\EpisodeDriverTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestEnv implements Environment
{
    protected $maxEpisodeSteps=5;
    protected $rewardThreshold=195.0;

    public function __construct(array $data)
    {
        $this->data = $data;
    }

    public function maxEpisodeSteps() : int
    {
        return $this->maxEpisodeSteps;
    }

    public function rewardThreshold() : float
    {
        return $this->rewardThreshold;
    }

    public function step($action) : array
    {
        return next($this->data);
    }

    public function reset() : mixed
    {
        [$obs,$reward,$done,$info] = reset($this->data);
        return $obs;
    }

    //public function legals($observation=null) : array
    //{}

    public function render(string $mode=null) : mixed
    {}

    public function close() : void
    {}

    public function seed(int $seed=null) : array
    {}

    public function show(bool $loop=null, int $delay=null) : mixed
    {}

    public function toString() : string
    {}

    public function enter() : void
    {}

    public function exit(Throwable $e=null) : bool
    {}
}

class TestAgent implements Agent
{
    public function __construct($assertActionObs,$actionResult,$assertUpdateLast)
    {
        $this->assertActionObs = $assertActionObs;
        $this->actionResult = $actionResult;
        $this->assertUpdateLast = $assertUpdateLast;
    }

    public function currents()
    {
        return [
            current($this->assertActionObs),
            current($this->actionResult),
            current($this->assertUpdateLast)
        ];
    }

    public function initialize() // : Operation
    {}

    public function policy()
    {}

    public function setElapsedTime($elapsedTime) : void
    {}

    public function action($observation,bool $traing)
    {
        if($observation != current($this->assertActionObs)) {
            echo "obs:";
            var_dump($observation);
            echo "actionObs:";
            var_dump(current($this->assertActionObs));
            throw new \Exception('invalid action observation');
        }
        next($this->assertActionObs);
        $action = current($this->actionResult);
        next($this->actionResult);
        return $action;
    }

    public function getQValue($observation) : float
    {
        return 1.0;
    }

    /**
    * @param iterable $experience
    */
    public function update($experience) : float
    {
        if($experience->last()!=current($this->assertUpdateLast)) {
            echo "exp::last:";
            var_dump($experience->last());
            echo "assertUpdateLast:";
            var_dump(current($this->assertUpdateLast));
            throw new \Exception('invalid update experience');
        }
        next($this->assertUpdateLast);

        return 1.0;
    }

    /**
    * @return bool $stepUpdate
    */
    public function isStepUpdate() : bool
    {
        return true;
    }

    public function startEpisode(int $episode) : void
    {}

    public function endEpisode(int $episode) : void
    {}

    /**
    * @return bool $stepUpdate
    */
    public function subStepLength() : int
    {
        return 1;
    }

    public function fileExists(string $filename) : bool
    {}

    public function saveWeightsToFile(string $filename) : void
    {}

    public function loadWeightsFromFile(string $filename) : void
    {}
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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testTrainBasic()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $experienceSize = 3;
        $batchSize = 1;
        $episodes = 2;
        $envdata = [
            //[$Obs,$reward,$done,$info]
            [100, null, null,   null],
            [101, 1,    false,  []],
            [102, 0.5,  false,  []],
            [103, 0,    true,   []],
        ];
        $env = new TestEnv($envdata);
        $assertActionObs =  [
            100, 101, 102,
            100, 101, 102,
        ];
        $actionResult    =  [
            10,   11,  12,
            20,   21,  22,
        ];
        $assertUpdateLast = [
            //[$observation,$action,$nextObs,$reward,$done,$info]
            [100, 10, 101, 1,   false, []],
            [101, 11, 102, 0.5, false, []],
            [102, 12, 103, 0,   true,  []],
            [100, 20, 101, 1,   false, []],
            [101, 21, 102, 0.5, false, []],
            [102, 22, 103, 0,   true,  []],
        ];
        $agent = new TestAgent($assertActionObs,$actionResult,$assertUpdateLast);
        $driver = new EpisodeDriver($la,$env, $agent, $experienceSize);
        $losses = $driver->train(numIterations:$episodes);
        $this->assertEquals([],$losses);
        $this->assertEquals([false,false,false],$agent->currents());
        $this->assertTrue(true);
    }

    public function testTrainWithEval()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $experienceSize = 3;
        $batchSize = 1;
        $episodes = 4;
        $evalInterval = 2;
        $numEvalEpisodes = 2;
        $envdata = [
            //[$Obs,$reward,$done, $info]
            [   100, null,  null,  null],
            [   101,  1,    false, []],
            [   102,  0.5,  false, []],
            [   103,  0,    true,  []],
        ];
        $env = new TestEnv($envdata);
        $assertActionObs =  [
            100, 101, 102,
            100, 101, 102,

            100, 101, 102, // eval
            100, 101, 102, // eval

            100, 101, 102,
            100, 101, 102,

            100, 101, 102, // eval
            100, 101, 102, // eval
        ];
        $actionResult = [
            10,   11,  12,
            20,   21,  22,

            110,  111, 112, // eval
            120,  121, 122, // eval

            30,   31,  32,
            40,   41,  42,

            130,  131, 132, // eval
            140,  141, 142, // eval
        ];
        $assertUpdateLast = [
            //[$obs,$action,$nextObs,$reward, $done, $info]
            [100, 10, 101, 1,   false, []],
            [101, 11, 102, 0.5, false, []],
            [102, 12, 103, 0,   true,  []],
            [100, 20, 101, 1,   false, []],
            [101, 21, 102, 0.5, false, []],
            [102, 22, 103, 0,   true,  []],

            [100, 30, 101, 1,   false, []],
            [101, 31, 102, 0.5, false, []],
            [102, 32, 103, 0,   true,  []],
            [100, 40, 101, 1,   false, []],
            [101, 41, 102, 0.5, false, []],
            [102, 42, 103, 0,   true,  []],
        ];
        $agent = new TestAgent($assertActionObs,$actionResult,$assertUpdateLast);
        $driver = new EpisodeDriver($la,$env, $agent, $experienceSize);
        $losses = $driver->train(
            numIterations:$episodes,
            evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
            metrics:['steps','reward','loss','val_steps','val_reward']
        );
        $this->assertEquals([
            'steps'     => [3,   3  ],
            'reward'    => [1.5, 1.5],
            'loss'      => [1.0, 1.0],
            'val_steps' => [3,   3  ],
            'val_reward'=> [1.5, 1.5],
        ],$losses);
        $this->assertEquals([false,false,false],$agent->currents());
        $this->assertTrue(true);
    }
}
