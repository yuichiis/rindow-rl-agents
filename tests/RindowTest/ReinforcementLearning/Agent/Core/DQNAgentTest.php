<?php
namespace RindowTest\ReinforcementLearning\Agent\Core\DQNAgentTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\ReinforcementLearning\Agent\Policy;
use Rindow\ReinforcementLearning\Env\Env;
use Rindow\ReinforcementLearning\Agent\Memory;
use Rindow\ReinforcementLearning\Agent\Core\DQNAgent;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestEnv implements Env
{
    public function __construct(array $data)
    {
        $this->data = $data;
    }

    public function step($action) : array
    {
        return next($this->data);
    }

    public function reset()
    {
        [$obs,$reward,$done,$info] = reset($this->data);
        return $obs;
    }

    public function legals($observation=null) : array
    {}

    public function render(string $mode=null)
    {}

    public function close() : void
    {}

    public function seed(int $seed=null) : array
    {}

    public function toString() : string
    {}

    public function enter() : void
    {}

    public function exit(Throwable $e=null) : bool
    {}
}

class TestPolicy implements Policy
{
    public function __construct($batchSize,$assertActionObs,$actionResult,$assertUpdateLast)
    {
        $this->batchSize = $batchSize;
        $this->assertActionObs = $assertActionObs;
        $this->actionResult = $actionResult;
        $this->assertUpdateLast = $assertUpdateLast;
    }
    public function initialize() // : Operation
    {}

    public function startEpisode(int $episode) : void
    {
        reset($this->assertActionObs);
        reset($this->actionResult);
        reset($this->assertUpdateLast);
    }

    public function endEpisode(int $episode) : void
    {}

    /**
    * @param Any $states
    * @return Any $action
    */
    public function action($observation)
    {
        if($observation != current($this->assertActionObs)) {
            var_dump($observation);
            throw new \Exception('invalid action observation');
        }
        next($this->assertActionObs);
        $action = current($this->actionResult);
        next($this->actionResult);
        return $action;
    }

    /**
    * @param iterable $experience
    */
    public function update($experience) : void
    {
        if($experience->last()!=current($this->assertUpdateLast)) {
            var_dump($experience->last());
            throw new \Exception('invalid update experience');
        }
        next($this->assertUpdateLast);
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
        return $this->batchSize;
    }

    /**
    * @return float $reward
    */
    public function customReward($stepCount,$observation,$reward,$done,$info) : float
    {
        return $reward;
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

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $experienceSize = 3;
        $batchSize = 1;
        $envdata = [
            //[$Obs,$reward,$done,$info]
            [100, null, null, null],
            [101, 1, false, []],
            [102, 2, false, []],
            [103, 0, true, []],
        ];
        $env = new TestEnv($envdata);
        $assertActionObs =  [100, 101, 102];
        $actionResult    =  [10,   11,  12];
        $assertUpdateLast = [
            //[$observation,$action,$nextObs,$reward,$done,$info]
            [100, 10, 101, 1, false, []],
            [101, 11, 102, 2, false, []],
            [102, 12, 103, 0, true,  []],
        ];
        $policy = new TestPolicy($batchSize,$assertActionObs,$actionResult,$assertUpdateLast);
        $agent = new DQNAgent($la,$env,$policy,$experienceSize);
        $agent->train($episode=1);
        $this->assertTrue(true);
    }
}
