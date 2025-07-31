<?php
namespace RindowTest\RL\Agents\Runner\ParallelStepRunnerTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment;
use Interop\Polite\AI\RL\Spaces\Space;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\ReplayBuffer;
use Rindow\RL\Agents\Runner\ParallelStepRunner;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestEnv implements Environment
{
    protected object $la;
    protected int $maxEpisodeSteps=200;
    protected float $rewardThreshold=195.0;
    protected array $data;
    protected bool $firstData = true;

    public function __construct(
        object $la,
        array $data
        )
    {
        $this->la = $la;
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
    public function observationSpace() : ?Space
    {}

    public function actionSpace() : ?Space
    {}


    public function step(mixed $action) : array
    {
        $la = $this->la;
        [$states,$reward,$done,$truncated,$info] = current($this->data);
        if(!in_array($la->scalar($action),$info['valid_directions'])) {
            echo "invalid action:";
            var_dump($action->toArray());
            echo "validActions:";
            var_dump($info['valid_directions']);
            throw new \InvalidArgumentException('invalid action');
        }
        [$states,$reward,$done,$truncated,$info] = next($this->data);
        $states = $la->array($states);
        return [$states,$reward,$done,$truncated,$info];
    }

    public function reset() : array
    {
        $la = $this->la;
        if($this->firstData) {
            $this->firstData = false;
            $data = reset($this->data);
        } else {
            $data = next($this->data);
        }
        [$states,$reward,$done,$truncated,$info] = $data;
        $states = $la->array($states);
        return [$states,$info];
    }

    //public function legals($state=null) : array
    //{}

    public function render(?string $mode=null) : mixed
    {}

    public function close() : void
    {}

    public function seed(?int $seed=null) : array
    {}

    public function show(?string $path=null,?bool $loop=null, ?int $delay=null) : mixed
    {}

    public function toString() : string
    {}

    public function enter() : void
    {}

    public function exit(?Throwable $e=null) : bool
    {}
}

class TestAgent implements Agent
{
    protected object $la;
    protected array $assertActionState;
    protected array $actionResult;
    protected array $assertUpdateLast;
    protected array $assertActionInfo;

    public function __construct(
        object $la,
        array $assertActionState,
        array $actionResult,
        array $assertUpdateLast,
        array $assertActionInfo,
        )
    {
        $this->la = $la;
        $this->assertActionState = $assertActionState;
        $this->actionResult = $actionResult;
        $this->assertUpdateLast = $assertUpdateLast;
        $this->assertActionInfo = $assertActionInfo;
    }

    public function register(?EventManager $eventManager=null) : void
    {}

    public function currents()
    {
        return [
            current($this->assertActionState),
            current($this->actionResult),
            current($this->assertUpdateLast)
        ];
    }

    public function initialize() : void // : Operation
    {}

    public function policy() : ?Policy
    {}

    public function setElapsedTime($elapsedTime) : void
    {}

    public function action(array|NDArray $state,?bool $training=null,?array $info=null) : NDArray
    {
        $la = $this->la;
        if(is_array($state)) {
            $newState = [];
            foreach($state as $st) {
                $newState[] = $la->scalar($la->squeeze($st));
            }
            $state = $newState;
        } else {
            $state = $la->scalar($state);
        }
        if($state != current($this->assertActionState)) {
            echo "state:";
            var_dump($state);
            echo "actionState:";
            var_dump(current($this->assertActionState));
            throw new \Exception('invalid action state');
        }
        next($this->assertActionState);
        if($info != current($this->assertActionInfo)) {
            echo "info:";
            var_dump($info);
            echo "actionInfo:";
            var_dump(current($this->assertActionInfo));
            throw new \Exception('invalid action info');
        }
        next($this->assertActionInfo);
        $action = current($this->actionResult);
        next($this->actionResult);
        $action = $la->array($action,dtype:NDArray::int32);
        return $action;
    }

    //public function maxQValue(mixed $state) : float
    //{
    //    return 1.0;
    //}

    /**
    * @param iterable $experience
    */
    public function update($experience) : float
    {
        $la = $this->la;
        $record = $experience->last();
        $record[0] = $la->scalar($record[0]); // $state
        $record[1] = $la->scalar($record[1]); // $action
        $record[2] = $la->scalar($record[2]); // $nextState
        if($record!=current($this->assertUpdateLast)) {
            echo "last:";
            var_dump($record);
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

    public function numRolloutSteps() : int
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

class ParallelStepRunnerTest extends TestCase
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
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
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
            //[$state,$reward,$done,$truncated,$info]
            [100, null, null,   null,  ['valid_directions'=>[10]]],
            [101, 1,    false,  false, ['valid_directions'=>[11]]],
            [102, 0.5,  false,  false, ['valid_directions'=>[12]]],
            [103, 1.5,  true,   false, ['valid_directions'=>[99]]],
            [104, null, null,   null,  ['valid_directions'=>[14]]],
            [105, 2.0,  false,  false, ['valid_directions'=>[15]]],
            [106, 2.5,  false,  false, ['valid_directions'=>[16]]],
        ];
        $envdata2 = [
            //[$state,$reward,$done,$truncated,$info]
            [200, null, null, null,   ['valid_directions'=>[20]]],
            [201, 21,   false,false,  ['valid_directions'=>[21]]],
            [202, 20.5, false,false,  ['valid_directions'=>[22]]],
            [203, 21.5, false,false,  ['valid_directions'=>[23]]],
            [204, 22.0, true, false,  ['valid_directions'=>[98]]],
            [205, null, null, null,   ['valid_directions'=>[24]]],
            //[206, 22.5, false,false,  ['valid_directions'=>[25]]],
        ];
        $envs = [];
        $envs[] = new TestEnv($la,$envdata1);
        $envs[] = new TestEnv($la,$envdata2);
        $assertActionState =  [
            [100,200],
            [101,201],
            [102,202],
            [104,203],
            [105,204],
            //[106,205],
        ];
        $actionResult    =  [
            $mo->array([10, 20,],),
            $mo->array([11, 21,],),
            $mo->array([12, 22,],),
            $mo->array([14, 23,],),
            $mo->array([15, 24,],),
            //$mo->array([16, 25,],),
        ];
        $assertUpdateLast = [
            //[$state,$action,$nextState,$reward,$done,$truncated,$info]
            [100, 10, 101, 1,    false, false, ['valid_directions'=>[11]]],
            [200, 20, 201, 21,   false, false, ['valid_directions'=>[21]]],
            [101, 11, 102, 0.5,  false, false, ['valid_directions'=>[12]]],
            [201, 21, 202, 20.5, false, false, ['valid_directions'=>[22]]],
            [102, 12, 103, 1.5,  true,  false, ['valid_directions'=>[99]]],
            [202, 22, 203, 21.5, false, false, ['valid_directions'=>[23]]],
            [104, 14, 105, 2.0,  false, false, ['valid_directions'=>[15]]],
            [203, 23, 204, 22.0, true,  false, ['valid_directions'=>[98]]],
            [105, 15, 106, 2.5,  false, false, ['valid_directions'=>[16]]],
        ];
        $assertActionInfo = [
            [['valid_directions'=>[10]],['valid_directions'=>[20]]],
            [['valid_directions'=>[11]],['valid_directions'=>[21]]],
            [['valid_directions'=>[12]],['valid_directions'=>[22]]],
            [['valid_directions'=>[99]],['valid_directions'=>[23]]],
        ];
        $agent = new TestAgent($la,$assertActionState,$actionResult,$assertUpdateLast,$assertActionInfo);
        $driver = new ParallelStepRunner($la,$envs, $agent, $experienceSize);
        $history = $driver->train(numIterations:$steps,
            verbose:0,logInterval:1,evalInterval:2,numEvalEpisodes:0,
            metrics:['steps','reward','loss'],
        );
        $this->assertEquals([
            'steps'  => [0,  2],
            'reward' => [0, 23.5],
            'loss'   => [1,  1],
        ],$history);
        [$actionStates,$actions,$updateLasts] = $agent->currents();
        $this->assertEquals([105,204],$actionStates);
        $this->assertEquals([15,24],$actions->toArray());
        $this->assertEquals([105,15,106,2.5,false,false,['valid_directions'=>[16]]],$updateLasts);
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
            //[$state,$reward,$done,$truncated, $info]
            [   100, null,  null,  null,  ['valid_directions'=>[10]]],
            [   101,  1,    false, false, ['valid_directions'=>[11]]],
            [   102,  0.5,  false, false, ['valid_directions'=>[12]]],
            [   103,  0,    true,  false, ['valid_directions'=>[99]]],
            [   104, null,  null,  null,  ['valid_directions'=>[14]]],
            [   105,  1,    false, false, ['valid_directions'=>[15]]],
        ];
        $envdata2 = [
            //[$state,$reward,$done,$truncated, $info]
            [   200, null,  null,  null,  ['valid_directions'=>[20]]],
            [   201,  1,    false, false, ['valid_directions'=>[21]]],
            [   202,  0.5,  false, false, ['valid_directions'=>[22]]],
            [   203,  0,    true,  false, ['valid_directions'=>[98]]],
            [   204, null,  null,  null,  ['valid_directions'=>[24]]],
            [   205,  1,    false, false, ['valid_directions'=>[25]]],
        ];
        $evalEnvdata = [
            //[$state,$reward,$done,$truncated, $info]
            [   300, null,  null,  null,  ['valid_directions'=>[30]]],
            [   301,  1,    false, false, ['valid_directions'=>[31]]],
            [   302,  0.5,  false, false, ['valid_directions'=>[32]]],
            [   303,  0,    true,  false, ['valid_directions'=>[97]]],
            [   400, null,  null,  null,  ['valid_directions'=>[40]]],
            [   401,  1,    false, false, ['valid_directions'=>[41]]],
            [   402,  0.5,  false, false, ['valid_directions'=>[42]]],
            [   403,  0,    true,  false, ['valid_directions'=>[96]]],
            [   500, null,  null,  null,  ['valid_directions'=>[50]]],
            [   501,  1,    false, false, ['valid_directions'=>[51]]],
            [   502,  0.5,  false, false, ['valid_directions'=>[52]]],
            [   503,  0,    true,  false, ['valid_directions'=>[96]]],
            [   600, null,  null,  null,  ['valid_directions'=>[60]]],
            [   601,  1,    false, false, ['valid_directions'=>[61]]],
            [   602,  0.5,  false, false, ['valid_directions'=>[62]]],
            [   603,  0,    true,  false, ['valid_directions'=>[95]]],
        ];
        $envs = [];
        $envs[] = new TestEnv($la,$envdata1);
        $envs[] = new TestEnv($la,$envdata2);
        $evalEnv = new TestEnv($la,$evalEnvdata);
        $assertActionState =  [
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
            30, 31, 32,
            40, 41, 42,
            [12, 22],
            [14, 24],
            50, 51, 52,
            60, 61, 62,
        ];
        $assertUpdateLast = [
            //[$states,$action,$nextState,$reward, $done,$truncated, $info]
            [   100,     10,     101,    1,   false, false, ['valid_directions'=>[11]]],
            [   200,     20,     201,    1,   false, false, ['valid_directions'=>[21]]],
            [   101,     11,     102,    0.5, false, false, ['valid_directions'=>[12]]],
            [   201,     21,     202,    0.5, false, false, ['valid_directions'=>[22]]],
            [   102,     12,     103,    0,    true, false, ['valid_directions'=>[99]]],
            [   202,     22,     203,    0,    true, false, ['valid_directions'=>[98]]],
            [   104,     14,     105,    1,   false, false, ['valid_directions'=>[15]]],
            [   204,     24,     205,    1,   false, false, ['valid_directions'=>[25]]],
        ];
        $assertActionInfo = [
            [['valid_directions'=>[10]],['valid_directions'=>[20]]],
            [['valid_directions'=>[11]],['valid_directions'=>[21]]],
            ['valid_directions'=>[30]],
            ['valid_directions'=>[31]],
            ['valid_directions'=>[32]],
            ['valid_directions'=>[40]],
            ['valid_directions'=>[41]],
            ['valid_directions'=>[42]],
            [['valid_directions'=>[12]],['valid_directions'=>[22]]],
            [['valid_directions'=>[99]],['valid_directions'=>[98]]],
            ['valid_directions'=>[50]],
            ['valid_directions'=>[51]],
            ['valid_directions'=>[52]],
            ['valid_directions'=>[60]],
            ['valid_directions'=>[61]],
            ['valid_directions'=>[62]],
        ];
        $agent = new TestAgent($la,$assertActionState,$actionResult,$assertUpdateLast,$assertActionInfo);
        $driver = new ParallelStepRunner($la,$envs, $agent, $experienceSize, evalEnv:$evalEnv);
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
