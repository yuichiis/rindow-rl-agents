<?php
namespace RindowTest\ReinforcementLearning\Agent\Policy\QNetworkTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\ReinforcementLearning\Agent\Selector;
use Rindow\ReinforcementLearning\Agent\Memory;
use Rindow\ReinforcementLearning\Agent\Policy\QNetwork;
use Rindow\ReinforcementLearning\Agent\Selector\AbstractSelector;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

use LogicException;
use InvalidArgumentException;


class TestMemory implements Memory
{
    public function __construct($data)
    {
        $this->data = $data;
    }

    public function size() : int
    {
        return count($this->data);
    }

    public function last()
    {
        $idx = array_key_last($this->data);
        return $this->data[$idx];
    }

    public function add($item) : void
    {
        $this->data[] = $item;
    }

    public function count()
    {
        return $this->size();
    }

    public function sample(int $quantity) : iterable
    {
        if($quantity!=count($this->data)) {
            throw new \Exception("unmatch datasize");
        }
        return $this->data;
    }
}

class TestModel
{
    public function __construct($la,$numActions,$assetPredict,$assertFit,$assertTrues)
    {
        $this->la = $la;
        $this->numActions = $numActions;
        $this->assetPredict = $assetPredict;
        $this->assertFit = $assertFit;
        $this->assertTrues = $assertTrues;
    }

    public function predict($value)
    {
        if($value->shape()!=$this->assetPredict->shape()) {
            throw new \Exception("unmatch predict input shape:".
                '['.implode(',',$value->shape()).']'.
                '. shoud be ['.implode(',',$this->assetPredict->shape()).']');
        }
        if($value->toArray()!=$this->assetPredict->toArray()) {
            var_dump($value->toArray());
            throw new \Exception("unmatch predict input data");
        }
        $batchSize =  count($value);
        $res = $this->la->alloc([$batchSize,$this->numActions]);
        $tmp = $res->reshape([$batchSize*$this->numActions]);
        for($i=0;$i<$batchSize*$this->numActions;$i++) {
            $tmp[$i] = $i;
        }
        return $res;
    }

    public function fit($inputs,$trues,$options)
    {
        if($inputs->shape()!=$this->assertFit->shape()) {
            throw new \Exception("unmatch fit input shape:".
                '['.implode(',',$inputs->shape()).']'.
                '. shoud be ['.implode(',',$this->assertFit->shape()).']');
        }
        if($inputs->toArray()!=$this->assertFit->toArray()) {
            var_dump($inputs->toArray());
            throw new \Exception("unmatch fit input data");
        }
        ///////////////////////////////////////////////////
        if($trues->shape()!=$this->assertTrues->shape()) {
            throw new \Exception("unmatch fit trues shape:".
                '['.implode(',',$trues->shape()).']'.
                '. shoud be ['.implode(',',$this->assertTrues->shape()).']');
        }
        if($trues->toArray()!=$this->assertTrues->toArray()) {
            var_dump($trues->toArray());
            throw new \Exception("unmatch fit trues data");
        }

        return ['loss'=>[0]];
    }

    public function loadWeights($modelWeights) : void
    {}
    public function saveWeights(&$modelWeights,$portable=null) : void
    {}
}

class TestSelector extends AbstractSelector
{
    public function __construct($assertValues,$returnAction)
    {
        $this->assertValues = $assertValues;
        $this->returnAction = $returnAction;
    }
    public function initialize()
    {}

    public function action($values,int $time=null) : int
    {
        $values = $this->getValues($values);
        if($values->toArray()!=$this->assertValues->toArray()) {
            var_dump($values->toArray());
            throw new \Exception("unmatch action values data");
        }
        return $this->returnAction;
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

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function testUpdateNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $obsSize = 4;
        $numActions = 2;
        $batchSize = 3;
        $gamma = 0.5;
        $assetcollectPredict = $la->array([
            [1,1,1,1],
            [2,2,2,2],
        ]);
        $assetPredict = $la->array([
            [1,0,0,0],
            [2,2,2,2],
            [1,1,1,1],
        ]);
        $assetTrues = $la->array([
            [0.5, 1],
            [1,   3],
            [4, 1.5],
        ]);
        $model = new TestModel($la,$numActions,$assetPredict,$assetPredict,$assetTrues);
        $collectModel = new TestModel($la,$numActions,$assetcollectPredict,null,null);
        $selector = new TestSelector(null,null);
        $qn = new QNetwork($la,
            $selector, $batchSize, $gamma,
            $obsSize, $numActions,
            null,null,null,
            $model, $collectModel
        );

        $obs0 = $la->array([1,0,0,0]);
        $obs1 = $la->array([1,1,1,1]);
        $obs2 = $la->array([2,2,2,2]);
        $memory = new TestMemory([
            //[$observation,$action,$nextObs,$reward,$done,$info]
            [$obs0, 0, $obs1, 0, false, []],
            [$obs2, 0,  null, 1, true,  []],
            [$obs1, 1, $obs2, 0, false, []],
        ]);
        $qn->update($memory);
        $this->assertTrue(true);
    }

    public function testUpdateNotEnough()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $obsSize = 4;
        $numActions = 2;
        $batchSize = 3;
        $gamma = 0.5;
        $collectModel = new TestModel($la,$numActions,null,null,null);
        $model = new TestModel($la,$numActions,null,null,null);
        $selector = new TestSelector(null,null);
        $qn = new QNetwork($la,
            $selector, $batchSize, $gamma,
            $obsSize, $numActions,
            null,null,null,
            $model, $collectModel
        );

        $obs0 = $la->array([1,0,0,0]);
        $obs1 = $la->array([1,1,1,1]);
        $memory = new TestMemory([
            //[$observation,$action,$nextObs,$reward,$done,$info]
            [$obs0, 0, $obs1, 0, false, []],
        ]);
        $qn->update($memory);
        $this->assertTrue(true);
    }

    public function testActionNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $obsSize = 4;
        $numActions = 2;
        $batchSize = 3;
        $gamma = 0.5;
        $assetPredict = $la->array([
            [11,22,33,44],
        ]);
        $collectModel = new TestModel($la,$numActions,null,null,null);
        $model = new TestModel($la,$numActions,$assetPredict,null,null);
        $assertValues = $la->array([0,1]);
        $returnAction = 1234;
        $selector = new TestSelector($assertValues,$returnAction);
        $qn = new QNetwork($la,
            $selector, $batchSize, $gamma,
            $obsSize, $numActions,
            null,null,null,
            $model, $collectModel
        );
        $observation = $la->array([11,22,33,44]);
        $action = $qn->action($observation);
        $this->assertEquals(1234,$action);
    }

    public function testBuildModel()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newNeuralNetworks($mo);
        $obsSize = 4;
        $numActions = 2;
        $batchSize = 3;
        $gamma = 0.5;
        $assetPredict = $la->array([
            [11,22,33,44],
        ]);
        $assertValues = $la->array([0,1]);
        $returnAction = 1234;
        $selector = new TestSelector($assertValues,$returnAction);
        $qn = new QNetwork($la,
            $selector, $batchSize, $gamma,
            $obsSize, $numActions,
            $nn
        );
        $this->assertTrue(true);
    }
}
