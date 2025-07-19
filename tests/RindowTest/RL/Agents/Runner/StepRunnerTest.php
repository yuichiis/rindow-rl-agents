<?php
namespace RindowTest\RL\Agents\Runner\StepRunnerTest;

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
use Rindow\RL\Agents\Runner\StepRunner;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestEnv implements Environment
{
    protected $la;
    protected $maxEpisodeSteps=200;
    protected $rewardThreshold=195.0;
    protected $data;

    public function __construct(
        $la,
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
        [$states,$reward,$done,$truncated,$info] = reset($this->data);
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
        $la,
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
        $state = $la->scalar($state);
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
            echo "exp::last:";
            var_dump($record);
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

class StepRunnerTest extends TestCase
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
        $envdata = [
            //[$state,$reward,$done,$truncated,$info]
            [100, null, null,   null,  ['valid_directions'=>[10]]],
            [101, 1,    false,  false, ['valid_directions'=>[11]]],
            [102, 0.5,  false,  false, ['valid_directions'=>[12]]],
            [103, 0,    true,   false, ['valid_directions'=>[20]]],
        ];
        $env = new TestEnv($la,$envdata);
        $assertActionState =  [100, 101, 102, 100];
        $actionResult    =  [10,   11,  12,  10];
        $assertUpdateLast = [
            //[$state,$action,$nextState,$reward,$done,$truncated,$info]
            [100, 10, 101, 1,   false, false, ['valid_directions'=>[11]]],
            [101, 11, 102, 0.5, false, false, ['valid_directions'=>[12]]],
            [102, 12, 103, 0,   true,  false, ['valid_directions'=>[20]]],
            [100, 10, 101, 1,   false, false, ['valid_directions'=>[11]]],
        ];
        $assertActionInfo = [
            ['valid_directions'=>[10]],
            ['valid_directions'=>[11]],
            ['valid_directions'=>[12]],
            ['valid_directions'=>[10]],
        ];
        $agent = new TestAgent($la,$assertActionState,$actionResult,$assertUpdateLast,$assertActionInfo);
        $driver = new StepRunner($la,$env, $agent, $experienceSize);
        $losses = $driver->train(numIterations:$steps);
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
        $steps = 4;
        $evalInterval = 2;
        $numEvalEpisodes = 2;
        $envdata = [
            //[$state,$reward,$done,$truncated, $info]
            [   100, null,  null,  null,  ['valid_directions'=>[10]]],
            [   101,  1,    false, false, ['valid_directions'=>[11]]],
            [   102,  0.5,  false, false, ['valid_directions'=>[12]]],
            [   103,  0,    true,  false, ['valid_directions'=>[30]]],
        ];
        $evalEnvdata = [
            //[$state,$reward,$done,$truncated, $info]
            [   200, null,  null,  null,  ['valid_directions'=>[20]]],
            [   201,  1,    false, false, ['valid_directions'=>[21]]],
            [   202,  0.5,  false, false, ['valid_directions'=>[22]]],
            [   203,  0,    true,  false, ['valid_directions'=>[40]]],
        ];
        $env = new TestEnv($la,$envdata);
        $evalEnv = new TestEnv($la,$evalEnvdata);
        $assertActionState =  [
            100, 101,
            200, 201, 202,
            200, 201, 202,
            102, 100,
            200, 201, 202,
            200, 201, 202,
        ];
        $actionResult = [
            10, 11,
            20, 21, 22,
            20, 21, 22,
            12, 10,
            20, 21, 22,
            20, 21, 22,
        ];
        $assertUpdateLast = [
            //[$states,$action,$nextState,$reward, $done,$truncated, $info]
            [   100,     10,     101,    1,   false, false, ['valid_directions'=>[11]]],
            [   101,     11,     102,    0.5, false, false, ['valid_directions'=>[12]]],
            [   102,     12,     103,    0,    true, false, ['valid_directions'=>[30]]],
            [   100,     10,     101,    1,   false, false, ['valid_directions'=>[11]]],
        ];
        $assertActionInfo = [
            ['valid_directions'=>[10]],
            ['valid_directions'=>[11]],

            ['valid_directions'=>[20]],
            ['valid_directions'=>[21]],
            ['valid_directions'=>[22]],
            ['valid_directions'=>[20]],
            ['valid_directions'=>[21]],
            ['valid_directions'=>[22]],

            ['valid_directions'=>[12]],
            ['valid_directions'=>[10]],

            ['valid_directions'=>[20]],
            ['valid_directions'=>[21]],
            ['valid_directions'=>[22]],
            ['valid_directions'=>[20]],
            ['valid_directions'=>[21]],
            ['valid_directions'=>[22]],
        ];
        $agent = new TestAgent($la,$assertActionState,$actionResult,$assertUpdateLast,$assertActionInfo);
        $driver = new StepRunner($la,$env, $agent, $experienceSize, evalEnv:$evalEnv);
        $losses = $driver->train(
            numIterations:$steps,
            evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
            metrics:['steps','reward','loss','val_steps','val_reward']
        );
        $this->assertEquals([
            'steps'     => [0,   3],
            'reward'    => [0,   1.5],
            'loss'      => [1.0, 1.0],
            'val_steps' => [3,   3],
            'val_reward'=> [1.5, 1.5],
        ],$losses);
        $this->assertEquals([false,false,false],$agent->currents());
        $this->assertTrue(true);
    }
}
