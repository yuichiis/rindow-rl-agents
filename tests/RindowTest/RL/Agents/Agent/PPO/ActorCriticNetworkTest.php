<?php
namespace RindowTest\RL\Agents\Agent\PPO\ActorCriticNetworkTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\ActorCriticNetwork;
use Rindow\RL\Gym\Core\Spaces\Box;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;


class ActorCriticNetworkTest extends TestCase
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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    public function testTrainingDiscrete()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $network = new ActorCriticNetwork($la,$nn,$stateShape=[1],$numActions=2,fcLayers:[100]);
        $lossFn = $nn->losses->Huber();
        $optimizer = $nn->optimizers->Adam();
        $trainableVariables = $network->trainableVariables();
        $states = $la->array([[0],[1]]);
        $nextQValues = $la->array([[0,1,2],[0,1,2]]);
        $concat = $nn->layers->Concatenate();
        for($i=0;$i<100;$i++) {
            $loss = $nn->with($tape=$g->GradientTape(), function()
                    use ($concat,$network,$lossFn,$states,$nextQValues) {
                [$action_out,$critic_out] = $network($states,true);
                $comb = $concat->forward([$action_out,$critic_out]);
                $loss = $lossFn->forward($nextQValues,$comb);
                return $loss;
            });
            $grads = $tape->gradient($loss,$trainableVariables);
            $optimizer->update($trainableVariables,$grads);
            $losses[] = $K->scalar($loss->value());
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('ActorCriticNetwork');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testGetQValuesDiscrete()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $network = new ActorCriticNetwork($la,$nn,$stateShape=[1],$numActions=2,fcLayers:[100]);
        $qValues = $network->getActionValues($la->array([[1.0]]));
        $this->assertEquals([1,2],$qValues->shape());   // (batches,numActions)

        $qValues2 = $network->getActionValues($la->array([[1.0],[1.0],[1.0]]));
        $this->assertEquals([3,2],$qValues2->shape());
        $qValues3 = $network->getActionValues($la->array([[1.0]]));
        $this->assertEquals([1,2],$qValues3->shape());
        $qValues4 = $network->getActionValues($la->array([[2.0]]));
        $this->assertEquals([1,2],$qValues4->shape());
        
        $this->assertEquals($qValues->toArray(),$qValues3->toArray());
        $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());

    }

    public function testTrainingContinuous()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $network = new ActorCriticNetwork($la,$nn,$stateShape=[1],$numActions=2,fcLayers:[100],continuous:true);
        $lossFn = $nn->losses->Huber();
        $optimizer = $nn->optimizers->Adam();
        $trainableVariables = $network->trainableVariables();
        $states = $la->array([[0],[1]]);
        $nextQValues = $la->array([[0,1,2],[0,1,2]]);
        $concat = $nn->layers->Concatenate();
        for($i=0;$i<100;$i++) {
            $loss = $nn->with($tape=$g->GradientTape(), function()
                    use ($g,$concat,$network,$lossFn,$states,$nextQValues) {
                [$action_out,$critic_out,$logStd] = $network($states,true);
                $action_out = $g->mul($logStd,$action_out);
                $comb = $concat->forward([$action_out,$critic_out]);
                $loss = $lossFn->forward($nextQValues,$comb);
                return $loss;
            });
            $grads = $tape->gradient($loss,$trainableVariables);
            $optimizer->update($trainableVariables,$grads);
            $losses[] = $K->scalar($loss->value());
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('ActorCriticNetwork');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testGetQValuesContinuous()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $network = new ActorCriticNetwork($la,$nn,$stateShape=[1],numActions:2,fcLayers:[100],continuous:true);
        $qValues = $network->getActionValues($la->array([[1.0]]));
        $this->assertEquals([1,2],$qValues->shape());   // (batches,numActions)

        $qValues2 = $network->getActionValues($la->array([[1.0],[1.0],[1.0]]));
        $this->assertEquals([3,2],$qValues2->shape());
        $qValues3 = $network->getActionValues($la->array([[1.0]]));
        $this->assertEquals([1,2],$qValues3->shape());
        $qValues4 = $network->getActionValues($la->array([[2.0]]));
        $this->assertEquals([1,2],$qValues4->shape());
        
        $this->assertEquals($qValues->toArray(),$qValues3->toArray());
        $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());

    }
}
