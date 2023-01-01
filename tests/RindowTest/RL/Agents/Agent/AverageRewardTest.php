<?php
namespace RindowTest\RL\Agents\Agent\AverageRewardTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Gym\ClassicControl\MultiarmedBandit\Slots;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\AverageReward;
use Rindow\RL\Agents\Network\Probabilities;
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
    public function __construct($fixedAction)
    {
        $this->fixedAction = $fixedAction;
    }

    public function register(EventManager $eventManager=null) : void
    {}

    public function initialize() // : Operation
    {}

    public function action(QPolicy $network, NDArray $state,bool $training) : NDArray
    {
        if($state!=1) {
            throw new \Exception('illegal state in policy:action');
        }
        return $this->fixedAction;
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

        $probabilities = [0.2, 0.4, 0.6, 0.9];
        $qtable = new Probabilities($la,$la->array([$probabilities]));
        $fixedActions = [0,1];
        foreach($fixedActions as $fixedAction) {
            $policy = new TestPolicy($fixedAction);
            $agent = new AverageReward($la,$qtable,$policy);
            for($i=0;$i<10;$i++) {
                $this->assertEquals($fixedAction,$agent->action(1,$training=true));
            }
        }
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $probabilities = [0.2, 0.4, 0.6, 0.9];

        $env = new Slots($la,$probabilities);
        $qtable = new Probabilities($la,$la->array([$probabilities]));
        $policy = new AnnealingEpsGreedy($la,$epsStart=0.9,$epsEnd=0.1,$decayRate=0.1);
        $agent = new AverageReward($la,$qtable,$policy);
        $driver = new EpisodeDriver($la,$env,$agent,$experienceSize=10000);

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
