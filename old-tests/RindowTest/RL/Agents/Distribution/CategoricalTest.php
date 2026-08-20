<?php
namespace RindowTest\RL\Agents\Distribution\CategoricalTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Variable;
use LogicException;
use InvalidArgumentException;
use Rindow\RL\Agents\Distribution\Categorical;

class CategoricalTest extends TestCase
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

    public function testSampleShape()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : None
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([0.5,0.5]));   // (numActions)
        $dist = new Categorical($nn, probs:$probs);
        $action = $dist->sample();
        $this->assertEquals([],$action->shape());       // ()
        $this->assertEquals(NDArray::int32,$action->dtype());
        $this->assertInstanceof(Variable::class,$action);

        // batchSize : 3
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([[0.5,0.5],[0.5,0.5],[0.5,0.5]])); // (batchSize,numActions)
        $dist = new Categorical($nn, probs:$probs);
        $action = $dist->sample();
        $this->assertEquals([3],$action->shape());                          // (batchSize)
        $this->assertEquals(NDArray::int32,$action->dtype());
        $this->assertInstanceof(Variable::class,$action);

        // batchSize : BatchShape(3)
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([0.5,0.5]));   // (numActions)
        $dist = new Categorical($nn, probs:$probs);
        $action = $dist->sample(batchShape:[3]);
        $this->assertEquals([3],$action->shape());       // ()
        $this->assertEquals(NDArray::int32,$action->dtype());
        $this->assertInstanceof(Variable::class,$action);

    }

    public function testLogProbShape()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : None
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([0.25,0.75]));   // (numActions)
        $action = $g->Variable($la->array(1,dtype:NDArray::int32));
        $dist = new Categorical($nn, probs:$probs);
        $logProb = $dist->logProb($action);
        $this->assertEquals([],$logProb->shape());      // ()
        $this->assertEquals(NDArray::float32,$logProb->dtype());
        $this->assertInstanceof(Variable::class,$logProb);

        // batchSize : 3
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([[0.25,0.75],[0.2,0.8],[0.5,0.5]])); // (batchSize,numActions)
        $action = $g->Variable($la->array([1,1,0],dtype:NDArray::int32));
        $dist = new Categorical($nn, probs:$probs);
        $logProb = $dist->logProb($action);
        $this->assertEquals([3],$logProb->shape());                          // (batchSize)
        $this->assertEquals(NDArray::float32,$logProb->dtype());
        $this->assertInstanceof(Variable::class,$logProb);

    }

    public function testEntropyShape()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : None
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([0.25,0.75]));   // (numActions)
        $dist = new Categorical($nn, probs:$probs);
        $entropy = $dist->entropy();
        $this->assertEquals([],$entropy->shape());      // ()
        $this->assertEquals(NDArray::float32,$entropy->dtype());
        $this->assertInstanceof(Variable::class,$entropy);

        // batchSize : 3
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([[0.25,0.75],[0.2,0.8],[0.5,0.5]])); // (batchSize,numActions)
        $dist = new Categorical($nn, probs:$probs);
        $entropy = $dist->entropy();
        $this->assertEquals([3],$entropy->shape());                          // (batchSize)
        $this->assertEquals(NDArray::float32,$entropy->dtype());
        $this->assertInstanceof(Variable::class,$entropy);

    }

    public function testSampleCompute()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : 3
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([[1,0,0],[0,1,0],[0,0,1]]));  // (batchSize,numActions)
        $target = $g->Variable($la->array([0,1,2],dtype:NDArray::int32));

        $action = $nn->with($tape=$g->GradientTape(),function () use ($nn,$probs) {
            $dist = new Categorical($nn, probs:$probs);
            $action = $dist->sample();
            return $action;
        });
        $this->assertEquals($target->toArray(),$action->toArray());

        $logits = $g->Variable($la->array([[0,-INF,-INF],[-INF,0,-INF],[-INF,-INF,0]]));  // (batchSize,numActions)
        $target = $g->Variable($la->array([0,1,2],dtype:NDArray::int32));
        $action = $nn->with($tape=$g->GradientTape(),function () use ($nn,$g,$logits) {
            $dist = new Categorical($nn, logits:$logits);
            $action = $dist->sample();
            return $action;
        });
        $this->assertEquals($target->toArray(),$action->toArray());
    }

    public function testlogProbCompute()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : 3
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([[1,1e-9,1e-9],[1e-9,1,1e-9],[1e-9,1e-9,1]]));  // (batchSize,numActions)
        $action = $g->Variable($la->array([0,1,2],dtype:NDArray::int32));
        $target = $g->Variable($la->array([1e-9,1e-9,1e-9]));
        $lossfunc = $nn->losses()->MeanSquaredError();

        [$logProb,$loss] = $nn->with($tape=$g->GradientTape(),function () use ($nn,$probs,$action,$target,$lossfunc) {
            $dist = new Categorical($nn, probs:$probs);
            $logProb = $dist->logProb($action);
            $loss = $lossfunc($target,$logProb);
            return [$logProb,$loss];
        });
        [$dProbs] = $tape->gradient($loss, [$probs]);
        //echo "logProb=".$la->toString($logProb)."\n";
        //echo "loss=".$la->toString($loss)."\n";
        //echo "dProbs=".$la->toString($dProbs)."\n";
        $this->assertTrue(true);

        // batchSize : 3
        // action space : Discrete(2)
        $logits = $g->Variable($la->array([[2,1e-9,1e-9],[1e-9,2,1e-9],[1e-9,1e-9,2]]));  // (batchSize,numActions)
        $action = $g->Variable($la->array([0,1,2],dtype:NDArray::int32));
        $target = $g->Variable($la->array([0,0,0]));
        $lossfunc = $nn->losses()->MeanSquaredError();

        [$logProb,$loss] = $nn->with($tape=$g->GradientTape(),function () use ($nn,$logits,$action,$target,$lossfunc) {
            $dist = new Categorical($nn, logits:$logits);
            $logProb = $dist->logProb($action);
            $loss = $lossfunc($target,$logProb);
            return [$logProb,$loss];
        });
        [$dLogits] = $tape->gradient($loss, [$logits]);
        //echo "logProb=".$la->toString($logProb)."\n";
        //echo "loss=".$la->toString($loss)."\n";
        //echo "dLogits=".$la->toString($dLogits)."\n";
        $truesLogProb = $la->array([-0.23954484, -0.23954484, -0.23954473]);
        $this->assertTrue($la->isclose($truesLogProb,$logProb));
        $truesdLogits = $la->array([
            [-0.0340176 ,  0.0170088 ,  0.0170088 ],
            [ 0.0170088 , -0.0340176 ,  0.0170088 ],
            [ 0.01700879,  0.01700879, -0.03401758],
        ]);
        $this->assertTrue($la->isclose($truesdLogits,$dLogits));
        $this->assertTrue(true);

    }

    public function testEntropyCompute()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : 3
        // action space : Discrete(2)
        $probs = $g->Variable($la->array([[1,1e-9,1e-9],[1e-9,1,1e-9],[1e-9,1e-9,1]]));  // (batchSize,numActions)
        $target = $g->Variable($la->array([1e-9,1e-9,1e-9]));
        $lossfunc = $nn->losses()->MeanSquaredError();

        [$entropy,$loss] = $nn->with($tape=$g->GradientTape(),function () use ($nn,$probs,$target,$lossfunc) {
            $dist = new Categorical($nn, probs:$probs);
            $entropy = $dist->entropy();
            $loss = $lossfunc($target,$entropy);
            return [$entropy,$loss];
        });
        [$dProbs] = $tape->gradient($loss, [$probs]);
        //echo "entropy=".$la->toString($entropy)."\n";
        //echo "loss=".$la->toString($loss)."\n";
        //echo "dProbs=".$la->toString($dProbs)."\n";
        $truesEntropy = $la->array([4.144658e-08, 4.144658e-08, 4.144658e-08]);
        $this->assertTrue($la->isclose($truesEntropy,$entropy));
        //$truesdProbs = $la->array([1e-7,1e-7,1e-7]);
        //$this->assertTrue($la->isclose($truesdProbs,$dProbs));
        //$this->assertTrue(true);

        // batchSize : 3
        // action space : Discrete(2)
        $logits = $g->Variable($la->array([[2,1e-9,1e-9],[1e-9,2,1e-9],[1e-9,1e-9,2]]));  // (batchSize,numActions)
        $target = $g->Variable($la->array([0,0,0]));
        $lossfunc = $nn->losses()->MeanSquaredError();

        [$entropy,$loss] = $nn->with($tape=$g->GradientTape(),function () use ($nn,$logits,$target,$lossfunc) {
            $dist = new Categorical($nn, logits:$logits);
            $entropy = $dist->entropy();
            $loss = $lossfunc($target,$entropy);
            return [$entropy,$loss];
        });
        [$dLogits] = $tape->gradient($loss, [$logits]);
        //echo "entropy=".$la->toString($entropy)."\n";
        //echo "loss=".$la->toString($loss)."\n";
        //echo "dLogits=".$la->toString($dLogits)."\n";
        $truesEntropy = $la->array([0.66557276248932,0.66557276248932,0.66557252407074]);
        $this->assertTrue($la->isclose($truesEntropy,$entropy));
        $truesdLogits = $la->array([
            [-0.14876795,  0.07438397,  0.07438397],
            [ 0.07438397, -0.14876795,  0.07438397],
            [ 0.07438397,  0.07438397, -0.14876795],
        ]);
        $this->assertTrue($la->isclose($truesdLogits,$dLogits));
    }

}
