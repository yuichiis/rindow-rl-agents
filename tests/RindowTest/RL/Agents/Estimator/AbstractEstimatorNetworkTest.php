<?php
namespace RindowTest\RL\Agents\Estimator\AbstractEstimatorNetworkTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Estimator\AbstractEstimatorNetwork;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestNetwork extends AbstractEstimatorNetwork
{
    protected object $model;
    protected array $actionShape;
    //protected $onesProb;
    //protected ?NDArray $masks=null;
    //protected ?NDArray $probabilities=null;

    public function __construct($builder,
            array $stateShape,
            int|array $actionShape,
            ?array $convLayers=null,?string $convType=null,?array $fcLayers=null,
            $activation=null,$kernelInitializer=null,
            ?array $outputOptions=null,
            //?NDArray $rules=null,
            ?object $model=null
        )
    {
        if(is_int($actionShape)) {
            $actionShape = [$actionShape];
        }
        parent::__construct($builder,$stateShape);
        $this->actionShape = $actionShape;

        $la = $this->la;
        if($model===null) {
            $model = $this->buildActorModel(
                $stateShape,$actionShape,
                $convLayers,$convType,$fcLayers,
                $activation,$kernelInitializer,
                $outputOptions,
            );
        }
        //$this->initializeRules($rules,$actionShape);
        $this->model = $model;
    }

    protected function buildActorModel(
        array $stateShape,
        array $actionShape,
        ?array $convLayers,
        ?string $convType,
        ?array $fcLayers,
        ?string $activation,
        ?string $kernelInitializer,
        ?array $outputOptions,
        )
    {
        $nn = $this->builder;
        $K = $this->backend;

        if($convLayers===null && $fcLayers===null) {
            $fcLayers = [32, 16];
        }

        $model = $this->buildMlpLayers(
            $stateShape,
            convLayers:$convLayers,
            convType:$convType,
            fcLayers:$fcLayers,
            activation:$activation,
            kernelInitializer:$kernelInitializer
        );

        $this->addOutputLayer(
            $model,
            $actionShape,
            $outputOptions,
        );

        return $model;
    }

}

class AbstractEstimatorNetworkTest extends TestCase
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

    public function testTrainingDiscreteActions()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $network = new TestNetwork($nn,stateShape:[1],actionShape:2,fcLayers:[100]);
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

    public function testTrainingContinuousActions()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $network = new TestNetwork($nn,stateShape:[1],actionShape:2,fcLayers:[100]);
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
        $plt->title('TestNetwork');
        $plt->show();
        $this->assertTrue(true);
    }

    //public function testSample()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $nn = $this->newBuilder($mo);
    //    $network = new TestNetwork($nn,$stateShape=[1],$actionShape=[4],fcLayers:[100]);
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
    //    $network = new TestNetwork($nn,$stateShape=[1],$actionShape=[4],fcLayers:[100],rules:$rules);
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
    //    $network = new TestNetwork($nn,stateShape:[1],actionShape:4,fcLayers:[100]);
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
    //    $network = new TestNetwork($nn,stateShape:[1],actionShape:4,fcLayers:[100],rules:$rules);
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
    //    $network = new TestNetwork($nn,stateShape:[],actionShape:2,fcLayers:[100]);
    //    $qValues = $network->getActionValues($la->array([[1.0]]));
    //    $this->assertEquals([1,2],$qValues->shape());
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
    //    $network = new TestNetwork($nn,stateShape:[],actionShape:4,fcLayers:[100],rules:$rules);
    //    $qValues = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues->shape());
    //    $qValues2 = $network->getActionValues($la->array([[1],[1],[1]],dtype:NDArray::int32));
    //    $this->assertEquals([3,4],$qValues2->shape());
    //    $qValues3 = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues3->shape());
    //    $qValues4 = $network->getActionValues($la->array([[2]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues4->shape());
//
    //    $la->nan2num($qValues,-INF);
    //    $la->nan2num($qValues3,-INF);
    //    $la->nan2num($qValues4,-INF);
    //    $this->assertEquals($qValues->toArray(),$qValues3->toArray());
    //    $this->assertNotEquals($qValues->toArray(),$qValues4->toArray());
//
    //}

    public function testGetQValues()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $network = new TestNetwork($nn,stateShape:[],actionShape:2,fcLayers:[100]);
        $qValues = $network->getActionValues($la->array([[1.0]]));
        $this->assertEquals([1,2],$qValues->shape());
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
