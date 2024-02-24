<?php
namespace RindowTest\RL\Agents\Network\ActorNetworkTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Network\ActorNetwork;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;


class ActorNetworkTest extends TestCase
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
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
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

        $network = new ActorNetwork($la,$nn,$obsSize=[1],$actionSize=[2],fcLayers:[100]);
        $lossFn = $nn->losses->Huber();
        $optimizer = $nn->optimizers->Adam();
        $trainableVariables = $network->trainableVariables();
        $states = $la->array([[0],[1]]);
        $nextQValues = $la->array([[0,1],[0,1]]);
        for($i=0;$i<100;$i++) {
            $loss = $nn->with($tape=$g->GradientTape(), function()
                    use ($network,$lossFn,$states,$nextQValues) {
                $qValues = $network($states,true);
                $loss = $lossFn->forward($nextQValues,$qValues);
                return $loss;
            });
            $grads = $tape->gradient($loss,$trainableVariables);
            $optimizer->update($trainableVariables,$grads);
            $losses[] = $K->scalar($loss->value());
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('ActorNetwork');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testSample()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $network = new ActorNetwork($la,$nn,$obsSize=[1],$actionSize=[4],fcLayers:[100]);
        $obs = $la->array([[1],[1],[1]],NDArray::uint32);
        $num = 1000;
        $actions = $la->alloc([$num,count($obs),1],NDArray::uint32);
        for($i=0;$i<$num;$i++) {
            $a = $network->sample($obs);
            $this->assertEquals([3,1],$a->shape());
            $la->copy($a,$actions[$i]);
        }
        $this->assertEquals(0,$la->min($la->astype($actions,dtype:NDArray::float32)));
        $this->assertEquals(3,$la->max($la->astype($actions,dtype:NDArray::float32)));
        $avg = $la->sum($actions)/$num/count($obs);
        $this->assertLessThan(1.6,$avg);
        $this->assertGreaterThan(1.4,$avg);

        // network with rules
        $rules = $la->array([
            [NAN,NAN,  1,  1],
            [  1,NAN,NAN,  1],
            [  1,  1,NAN,NAN],
        ]);
        $network = new ActorNetwork($la,$nn,$obsSize=[1],$actionSize=[4],fcLayers:[100],rules:$rules);
        $obs = $la->array([[0],[1],[2]],NDArray::uint32);
        $num = 1000;
        $actions = $la->alloc([$num,count($obs),1],NDArray::uint32);
        for($i=0;$i<$num;$i++) {
            $a = $network->sample($obs);
            $this->assertEquals([3,1],$a->shape());
            $la->copy($a,$actions[$i]);
        }
        $this->assertEquals(0,$la->min($la->astype($actions,dtype:NDArray::float32)));
        $this->assertEquals(3,$la->max($la->astype($actions,dtype:NDArray::float32)));
        $avg = $la->sum($actions)/$num/count($obs);
        $this->assertLessThan(1.6,$avg);
        $this->assertGreaterThan(1.4,$avg);
    }

    public function testGetQValues()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $network = new ActorNetwork($la,$nn,$obsSize=[1],$actionSize=[2],fcLayers:[100]);
        $qValues = $network->getQValues($la->array([[1.0]]));
        $this->assertEquals([1,2],$qValues->shape());
        $qValues2 = $network->getQValues($la->array([[1.0],[1.0],[1.0]]));
        $this->assertEquals([3,2],$qValues2->shape());
        $qValues3 = $network->getQValues($la->array([[1.0]]));
        $this->assertEquals([1,2],$qValues3->shape());
        $qValues4 = $network->getQValues($la->array([[2.0]]));
        $this->assertEquals([1,2],$qValues4->shape());
        
        $this->assertEquals($qValues->toArray(),$qValues3->toArray());
        $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());

        // network with rules
        $rules = $la->array([
            [NAN,NAN,  1,  1],
            [  1,NAN,NAN,  1],
            [  1,  1,NAN,NAN],
        ]);
        $network = new ActorNetwork($la,$nn,$obsSize=[1],$actionSize=[4],fcLayers:[100],rules:$rules);
        $qValues = $network->getQValues($la->array([[1.0]]));
        $this->assertEquals([1,4],$qValues->shape());
        $qValues2 = $network->getQValues($la->array([[1.0],[1.0],[1.0]]));
        $this->assertEquals([3,4],$qValues2->shape());
        $qValues3 = $network->getQValues($la->array([[1.0]]));
        $this->assertEquals([1,4],$qValues3->shape());
        $qValues4 = $network->getQValues($la->array([[2.0]]));
        $this->assertEquals([1,4],$qValues4->shape());
        
        $this->assertEquals($qValues->toArray(),$qValues3->toArray());
        $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());

    }
}
