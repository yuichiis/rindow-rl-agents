<?php
namespace RindowTest\RL\Agents\Agent\AverageRewardTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Gym\ClassicControl\MultiarmedBandit\Slots;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\AverageReward\AverageReward;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestPolicy implements Policy
{
    protected NDArray $fixedActionsOutput;
    
    public function __construct($fixedActionsOutput)
    {
        $this->fixedActionsOutput = $fixedActionsOutput;
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    public function register(?EventManager $eventManager=null) : void
    {}

    public function initialize() : void // : Operation
    {}

    public function actions(Estimator $network, NDArray $state, bool $training, ?NDArray $masks) : NDArray
    {
        if($state->shape()==[1,1] && $state[0][0]!=0) {
            throw new \Exception('illegal state in policy:action');
        }
        return $this->fixedActionsOutput;
    }
}

class AverageRewardTest extends TestCase
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

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();

        $probabilities = [0.2, 0.4, 0.6, 0.9];
        $fixedActions = $la->array([0,1],NDArray::int32);
        $state = $la->array([0],NDArray::int32);
        foreach($fixedActions as $fixedAction) {
            $fixedActionsOutput = $la->array([$fixedAction],dtype:NDArray::int32);
            $policy = new TestPolicy($fixedActionsOutput);
            $agent = new AverageReward($la,numActions:count($probabilities),policy:$policy);
            for($i=0;$i<10;$i++) {
                $action = $agent->action($state,training:true);
                //echo "fixedActionsOutput(".implode(',',$fixedActionsOutput->shape()).")=".$mo->toString($fixedActionsOutput)."\n";
                //echo "action(".implode(',',$action->shape()).")=".$mo->toString($action)."\n";
                $action = $la->scalar($action);
                $this->assertEquals($fixedAction,$action);
            }
        }
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $stateFunc = function($env,$x,$done) use ($la) {
            return $la->expandDims($x,axis:-1);
        };
        $plt = new Plot($this->getPlotConfig(),$mo);
        $probabilities = [0.2, 0.4, 0.6, 0.9];

        $env = new Slots($la,$probabilities);
        $policy = new AnnealingEpsGreedy($la,$epsStart=0.9,$epsEnd=0.1,$decayRate=0.1);
        $agent = new AverageReward($la,numActions:count($probabilities),policy:$policy);
        $driver = new EpisodeDriver($la,$env,$agent,$experienceSize=10000);
        $driver->setCustomStateFunction($stateFunc);

        $numIterations = 50;
        $evalInterval  =  1;
        $numEvalEpisodes= 1000;
        $history = $driver->train(
            numIterations:$numIterations, metrics:['reward','loss','val_reward'],
            evalInterval:$evalInterval, numEvalEpisodes:$numEvalEpisodes, verbose:0);
        $ep = $mo->arange((int)($numIterations/$evalInterval)+1,0,$evalInterval);
        $plt->plot($ep,$la->array(array_merge([0.5],$history['val_reward'])))[0];
        $plt->legend(['val_reward']);
        $plt->title('AverageReward');
        $plt->show();
        $this->assertTrue(true);
    }
}
