<?php
namespace RindowTest\RL\Agents\Agent\UCB1\UCB1Test;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Gym\ClassicControl\MultiarmedBandit\Slots;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Agent\UCB1\UCB1;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class UCB1Test extends TestCase
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

    public function testAction()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $probabilities = [0.2, 0.4, 0.6, 0.9];
        //$valueTable = new ValueTable($la,$la->array([$probabilities]));
        $fixedActions = [0];
        $state = $la->array(1,NDArray::int32);
        foreach($fixedActions as $fixedAction) {
            $agent = new UCB1($la,count($probabilities));
            for($i=0;$i<10;$i++) {
                $action = $agent->action($state,training:true);
                $action = $la->scalar($action);
                $this->assertEquals($fixedAction,$action);
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
        //$valueTable = new ValueTable($la,$la->array([$probabilities]));
        $policy = new AnnealingEpsGreedy($la,$epsStart=0.9,$epsEnd=0.1,$decayRate=0.1);
        $agent = new UCB1($la,count($probabilities),mo:$mo);
        $driver = new EpisodeRunner($la,$env,$agent,$experienceSize=10000);

        $numIterations = 2500;
        $evalInterval  =  100;
        $numEvalEpisodes= 0;
        $history = $driver->train(
            numIterations:$numIterations, metrics:['reward','loss','val_reward'],
            evalInterval:$evalInterval, numEvalEpisodes:$numEvalEpisodes, verbose:0);
        $ep = $mo->arange((int)($numIterations/$evalInterval)+1,0,$evalInterval);
        $plt->plot($ep,$la->array(array_merge([0.5],$history['reward'])))[0];
        $plt->legend(['reward']);
        $plt->title('UCB1');
        $plt->show();
        $this->assertTrue(true);
    }
}
