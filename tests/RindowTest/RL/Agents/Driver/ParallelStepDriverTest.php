<?php
namespace RindowTest\RL\Agents\Driver\ParallelStepDriverTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Driver\ParallelStepDriver;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestEnv implements Environment
{
    protected $maxEpisodeSteps=200;
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
        $results = current($this->data);
        next($this->data);
        return $results;
    }

    public function reset() : mixed
    {
        [$obs,$reward,$done,$info] = current($this->data);
        next($this->data);
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

    public function register(EventManager $eventManager=null) : void
    {}

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

    public function action(mixed $observation,bool $traing) : mixed
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

    public function maxQValue(mixed $observation) : float
    {
        return 1.0;
    }

    /**
    * @param iterable $experience
    */
    public function update($experience) : float
    {
        if($experience->last()!=current($this->assertUpdateLast)) {
            echo "last:";
            var_dump($experience->last());
            echo "updateLast:";
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

class ParallelStepDriverTest extends TestCase
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
            'renderer.execBackground' => true,
        ];
    }

    public function testTrainBasic()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $experienceSize = 3;
        $batchSize = 1;
        $steps = 4;
        $envdata1 = [
            //[$Obs,$reward,$done,$info]
            [100, null, null,   null],
            [101, 1,    false,  []],
            [102, 0.5,  false,  []],
            [103, 1.5,  true,   []],
            [104, null, null,   null],
            [105, 2.0,  false,  []],
            [106, 2.5,  false,  []],
        ];
        $envdata2 = [
            //[$Obs,$reward,$done,$info]
            [200, null, null, null],
            [201, 21,   false,  []],
            [202, 20.5, false,  []],
            [203, 21.5, false,  []],
            [204, 22.0, true,   []],
            [205, null, null, null],
            [206, 22.5, false,  []],
        ];
        $envs = [];
        $envs[] = new TestEnv($envdata1);
        $envs[] = new TestEnv($envdata2);
        $assertActionObs =  [
            [100,200],
            [101,201],
            [102,202],
            [104,203],
            [105,204],
            [106,205],
        ];
        $actionResult    =  [
            $mo->array([10, 20,],),
            $mo->array([11, 21,],),
            $mo->array([12, 22,],),
            $mo->array([14, 23,],),
            $mo->array([15, 24,],),
            $mo->array([16, 25,],),
        ];
        $assertUpdateLast = [
            //[$observation,$action,$nextObs,$reward,$done,$info]
            [100, 10, 101, 1,    false, []],
            [200, 20, 201, 21,   false, []],
            [101, 11, 102, 0.5,  false, []],
            [201, 21, 202, 20.5, false, []],
            [102, 12, 103, 1.5,  true,  []],
            [202, 22, 203, 21.5, false, []],
            [104, 14, 105, 2.0,  false, []],
            [203, 23, 204, 22.0, true,  []],
            [105, 15, 106, 2.5,  false, []],
            [205, 25, 206, 22.5, false, []],
        ];
        $agent = new TestAgent($assertActionObs,$actionResult,$assertUpdateLast);
        $driver = new ParallelStepDriver($la,$envs, $agent, $experienceSize);
        $history = $driver->train(numIterations:$steps,
            verbose:2,logInterval:1,evalInterval:2,numEvalEpisodes:0,
            metrics:['steps','reward','loss'],
        );
        $this->assertEquals([
            'steps'  => [0,  2],
            'reward' => [0, 23.5],
            'loss'   => [1,  1],
        ],$history);
        [$actionObss,$actions,$updateLasts] = $agent->currents();
        $this->assertEquals([105,204],$actionObss);
        $this->assertEquals([15,24],$actions->toArray());
        $this->assertEquals([105,15,106,2.5,false,[]],$updateLasts);
        $this->assertTrue(true);
    }

    public function testTrainWithEval()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $experienceSize = 3;
        $batchSize = 1;
        $steps = 4;
        $evalInterval = 2;
        $numEvalEpisodes = 2;
        $envdata1 = [
            //[$Obs,$reward,$done, $info]
            [   100, null,  null,  null],
            [   101,  1,    false, []],
            [   102,  0.5,  false, []],
            [   103,  0,    true,  []],
            [   104, null,  null,  null],
            [   105,  1,    false, []],
        ];
        $envdata2 = [
            //[$Obs,$reward,$done, $info]
            [   200, null,  null,  null],
            [   201,  1,    false, []],
            [   202,  0.5,  false, []],
            [   203,  0,    true,  []],
            [   204, null,  null,  null],
            [   205,  1,    false, []],
        ];
        $evalEnvdata = [
            //[$Obs,$reward,$done, $info]
            [   300, null,  null,  null],
            [   301,  1,    false, []],
            [   302,  0.5,  false, []],
            [   303,  0,    true,  []],
            [   400, null,  null,  null],
            [   401,  1,    false, []],
            [   402,  0.5,  false, []],
            [   403,  0,    true,  []],
            [   500, null,  null,  null],
            [   501,  1,    false, []],
            [   502,  0.5,  false, []],
            [   503,  0,    true,  []],
            [   600, null,  null,  null],
            [   601,  1,    false, []],
            [   602,  0.5,  false, []],
            [   603,  0,    true,  []],
        ];
        $envs = [];
        $envs[] = new TestEnv($envdata1);
        $envs[] = new TestEnv($envdata2);
        $evalEnv = new TestEnv($evalEnvdata);
        $assertActionObs =  [
            [100, 200],
            [101, 201],
            300, 301, 302,
            400, 401, 402,
            [102, 202], 
            [104, 204],
            500, 501, 502,
            600, 601, 602,
        ];
        $actionResult = [
            [10, 20],
            [11, 21],
            20, 21, 22,
            20, 21, 22,
            [12, 22],
            [20, 10],
            30, 31, 32,
            30, 31, 32,
        ];
        $assertUpdateLast = [
            //[$obs,$action,$nextObs,$reward, $done, $info]
            [   100,     10,     101,    1,   false, []],
            [   200,     20,     201,    1,   false, []],
            [   101,     11,     102,    0.5, false, []],
            [   201,     21,     202,    0.5, false, []],
            [   102,     12,     103,    0,    true, []],
            [   202,     22,     203,    0,    true, []],
            [   104,     20,     105,    1,   false, []],
            [   204,     10,     205,    1,   false, []],
        ];
        $agent = new TestAgent($assertActionObs,$actionResult,$assertUpdateLast);
        $driver = new ParallelStepDriver($la,$envs, $agent, $experienceSize, evalEnv:$evalEnv);
        $losses = $driver->train(
            numIterations:$steps,
            evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
            metrics:['steps','reward','loss','val_steps','val_reward']
        );
        $this->assertEquals([
            'steps'     => [0,   2],
            'reward'    => [0,   1.0],
            'loss'      => [1.0, 1.0],
            'val_steps' => [3,   3],
            'val_reward'=> [1.5, 1.5],
        ],$losses);
        $this->assertEquals([false,false,false],$agent->currents());
        $this->assertTrue(true);
    }
}
