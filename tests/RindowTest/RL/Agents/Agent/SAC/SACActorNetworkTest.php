<?php
namespace RindowTest\RL\Agents\Agent\SAC\SACActorNetworkTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\SAC\ActorNetwork;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;


class SACActorNetworkTest extends TestCase
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

    public function testTraining()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $actionMin = $la->array([-2,-2]);
        $actionMax = $la->array([2,2]);
        $network = new ActorNetwork(
            builder:$nn,
            stateShape:[1],
            numActions:2,
            fcLayers:[100],
            actionMin:$actionMin,
            actionMax:$actionMax,
        );
        $lossFn = $nn->losses->Huber();
        $optimizer = $nn->optimizers->Adam();
        $trainableVariables = $network->trainableVariables();
        $states = $la->array([[0],[1]]);
        $targetActions = $la->array([[0,1],[0,1]]);
        $targetLogStd = $la->array([[0],[0]]);
        for($i=0;$i<100;$i++) {
            $loss = $nn->with($tape=$g->GradientTape(), function()
                    use ($g,$network,$lossFn,$states,$targetActions,$targetLogStd) {
                [$actorActions,$logStd] = $network($states,true);
                $actorLoss = $lossFn->forward($targetActions,$actorActions);
                $logStdLoss = $lossFn->forward($targetLogStd,$logStd);
                $loss = $g->add($actorLoss,$logStdLoss);
                return $loss;
            });
            $grads = $tape->gradient($loss,$trainableVariables);
            $optimizer->update($trainableVariables,$grads);
            $losses[] = $K->scalar($loss->value());
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('ActorNetwork for SAC');
        $plt->show();
        $this->assertTrue(true);
    }

    //public function testSample()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
    //    $network = new QNetwork($la,$nn,$stateShape=[1],$numActions=4,fcLayers:[100]);
    //    $states = $la->array([[1],[1],[1]],NDArray::int32);
    //    $num = 1000;
    //    $actions = $la->alloc([$num,count($states),1],NDArray::int32);
    //    for($i=0;$i<$num;$i++) {
    //        $a = $network->sample($states);
    //        $this->assertEquals([3,1],$a->shape());
    //        $la->copy($a,$actions[$i]);
    //    }
    //    $this->assertEquals(0,$la->min($actions));
    //    $this->assertEquals(3,$la->max($actions));
    //    $avg = $la->sum($actions)/$num/count($states);
    //    $this->assertLessThan(1.6,$avg);
    //    $this->assertGreaterThan(1.4,$avg);
//
    //    // network with rules
    //    $rules = $la->array([
    //        [NAN,NAN,  1,  1],
    //        [  1,NAN,NAN,  1],
    //        [  1,  1,NAN,NAN],
    //    ]);
    //    $network = new QNetwork($la,$nn,$stateShape=[1],$numActions=4,fcLayers:[100],rules:$rules);
    //    $states = $la->array([[0],[1],[2]],NDArray::int32);
    //    $num = 1000;
    //    $actions = $la->alloc([$num,count($states),1],NDArray::int32);
    //    for($i=0;$i<$num;$i++) {
    //        $a = $network->sample($states);
    //        $this->assertEquals([3,1],$a->shape());
    //        $la->copy($a,$actions[$i]);
    //    }
    //    $this->assertEquals(0,$la->min($actions));
    //    $this->assertEquals(3,$la->max($actions));
    //    $avg = $la->sum($actions)/$num/count($states);
    //    $this->assertLessThan(1.6,$avg);
    //    $this->assertGreaterThan(1.4,$avg);
    //}

    //public function testProbabilities()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
    //
    //
    //    // no rules
    //    $network = new QNetwork($la,$nn,stateShape:[1],numActions:4,fcLayers:[100]);
    //    $states = $la->array([1,0,2],NDArray::int32);    // (batches=3,state=1)
    //    $probs = $network->probabilities($states);
    //    //$this->assertEquals([3,4],$probs->shape());
    //    //$this->assertEquals([
    //    //    [0.25,0.25,0.25,0.25],
    //    //    [0.25,0.25,0.25,0.25],
    //    //    [0.25,0.25,0.25,0.25],
    //    //],$probs->toArray());
    //    $this->assertNull($probs);
    //
    //    // with rules
    //    $rules = $la->array([
    //        [NAN,NAN,  1,  1],
    //        [  1,NAN,NAN,  1],
    //        [  1,  1,NAN,NAN],
    //        [NAN,  1,  1,NAN],
    //    ]);
    //    $network = new QNetwork($la,$nn,stateShape:[1],numActions:4,fcLayers:[100],rules:$rules);
    //    $states = $la->array([[1],[0],[2]],NDArray::int32);    // (batches=3,state=1)
    //    $probs = $network->probabilities($states);
    //    $this->assertEquals([3,4],$probs->shape());
    //    $this->assertEquals([
    //        [0.5 ,0.0 ,0.0 ,0.5 ],
    //        [0.0 ,0.0 ,0.5 ,0.5 ],
    //        [0.5 ,0.5 ,0.0 ,0.0 ],
    //    ],$probs->toArray());
    //}

    //public function testGetQValues()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
//
    //    $network = new QNetwork($la,$nn,$stateShape=[1],$numActions=2,fcLayers:[100]);
    //    $qValues = $network->getActionValues($la->array([[1.0]]));
    //    $this->assertEquals([1,2],$qValues->shape());   // (batches,numActions)
    //    $qValues2 = $network->getActionValues($la->array([[1.0],[1.0],[1.0]]));
    //    $this->assertEquals([3,2],$qValues2->shape());
    //    $qValues3 = $network->getActionValues($la->array([[1.0]]));
    //    $this->assertEquals([1,2],$qValues3->shape());
    //    $qValues4 = $network->getActionValues($la->array([[2.0]]));
    //    $this->assertEquals([1,2],$qValues4->shape());
    //    
    //    $this->assertEquals($qValues->toArray(),$qValues3->toArray());
    //    $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());
//
    //    // network with rules
    //    $rules = $la->array([
    //        [NAN,NAN,  1,  1],
    //        [  1,NAN,NAN,  1],
    //        [  1,  1,NAN,NAN],
    //    ]);
    //    $network = new QNetwork($la,$nn,$stateShape=[1],$numActions=4,fcLayers:[100],rules:$rules);
    //    $qValues = $network->getActionValues($la->array([[1.0]]));
    //    $this->assertEquals([1,4],$qValues->shape());
    //    $qValues2 = $network->getActionValues($la->array([[1.0],[1.0],[1.0]]));
    //    $this->assertEquals([3,4],$qValues2->shape());
    //    $qValues3 = $network->getActionValues($la->array([[1.0]]));
    //    $this->assertEquals([1,4],$qValues3->shape());
    //    $qValues4 = $network->getActionValues($la->array([[2.0]]));
    //    $this->assertEquals([1,4],$qValues4->shape());
    //    
    //    $this->assertEquals($qValues->toArray(),$qValues3->toArray());
    //    $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());
//
    //}

    public function testGetQValues()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $actionMin = $la->array([-2,-2]);
        $actionMax = $la->array([2,2]);
        $network = new ActorNetwork(
            builder:$nn,
            stateShape:[1],
            numActions:2,
            fcLayers:[100],
            actionMin:$actionMin,
            actionMax:$actionMax,
        );
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

        // =============== with std =================
        $qValues = $network->getActionValues($la->array([[1.0]]),std:true);
        $this->assertEquals([1,2],$qValues[0]->shape());   // (batches,numActions)
        $this->assertEquals([1,1],$qValues[1]->shape());   // (batches,1)
        $qValues2 = $network->getActionValues($la->array([[1.0],[1.0],[1.0]]),std:true);
        $this->assertEquals([3,2],$qValues2[0]->shape());
        $this->assertEquals([3,1],$qValues2[1]->shape());
        $qValues3 = $network->getActionValues($la->array([[1.0]]),std:true);
        $this->assertEquals([1,2],$qValues3[0]->shape());
        $this->assertEquals([1,1],$qValues3[1]->shape());
        $qValues4 = $network->getActionValues($la->array([[2.0]]),std:true);
        $this->assertEquals([1,2],$qValues4[0]->shape());
        $this->assertEquals([1,1],$qValues4[1]->shape());
        
        $this->assertEquals($qValues[0]->toArray(),$qValues3[0]->toArray());
        $this->assertEquals($qValues[1]->toArray(),$qValues3[1]->toArray());
        $this->assertNotEquals($qValues[0]->toArray(),$qValues4[0]->toArray());
        $this->assertNotEquals($qValues[1]->toArray(),$qValues4[1]->toArray()); // log_prob includes noise
        
    }
}
