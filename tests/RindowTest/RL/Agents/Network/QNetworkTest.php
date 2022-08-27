<?php
namespace RindowTest\RL\Agents\Network\QNetworkTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\AI\RL\Environment;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Network\QNetwork;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;


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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
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

        $network = new QNetwork($la,$nn,$obsSize=[1],$numActions=2,fcLayers:[100]);
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
        $plt->title('QNetwork');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testSample()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $network = new QNetwork($la,$nn,$obsSize=[1],$numActions=2,fcLayers:[100]);
        $obs = 1;
        $actions = [];
        $num = 1000;
        for($i=0;$i<$num;$i++) {
            $actions[] = $network->sample($obs);
        }
        $actions = $la->array($actions);
        $this->assertEquals(0,$la->min($actions));
        $this->assertEquals(1,$la->max($actions));
        $avg = $la->sum($actions)/$num;
        $this->assertLessThan(0.6,$avg);
        $this->assertGreaterThan(0.4,$avg);
    }

    public function testGetQValues()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $network = new QNetwork($la,$nn,$obsSize=[1],$numActions=2,fcLayers:[100]);
        $qValues = $network->getQValues(1);
        $this->assertEquals([2],$qValues->shape());
        $qValues2 = $network->getQValues([1]);
        $this->assertEquals([1,2],$qValues2->shape());
        $qValues3 = $network->getQValues($la->array([1]));
        $this->assertEquals([2],$qValues3->shape());
        $qValues4 = $network->getQValues(2);
        $this->assertEquals([2],$qValues4->shape());
        
        $this->assertEquals($qValues->toArray(),$qValues2->toArray()[0]);
        $this->assertEquals($qValues->toArray(),$qValues3->toArray());
        $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());
    }
}
