<?php
namespace RindowTest\RL\Agents\Distribution\NormalTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Variable;
use LogicException;
use InvalidArgumentException;
use Rindow\RL\Agents\Distribution\Normal;

class NormalTest extends TestCase
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
        // action space : Box(2)
        $mean = $g->Variable($la->array([4,3]));    // (numActions)
        $std  = $g->Variable($la->array([2,1]));    // (numActions)
        $dist = new Normal($nn,$mean,$std);
        $action = $dist->sample();
        $this->assertEquals([2],$action->shape());
        $this->assertEquals(NDArray::float32,$action->dtype());
        $this->assertInstanceof(Variable::class,$action);
        $action2 = $dist->sample();
        //var_dump($action);
        $this->assertFalse($la->isclose(
            $g->ndarray($action),
            $g->ndarray($action2)
        ));

        // batchSize : 3
        // action space : Box(1)
        $mean = $g->Variable($la->array([[5],[4],[3]]));    // (batchSize,numActions)
        $std  = $g->Variable($la->array([[3],[2],[1]]));    // (batchSize,numActions)
        $dist = new Normal($nn,$mean,$std);
        $action = $dist->sample();
        $this->assertEquals([3,1],$action->shape());        // (batchSize,numActions)
        $this->assertEquals(NDArray::float32,$action->dtype());
        $this->assertInstanceof(Variable::class,$action);
        $action2 = $dist->sample();
        //var_dump($action);
        $this->assertFalse($la->isclose(
            $g->ndarray($action),
            $g->ndarray($action2)
        ));

        // batchSize : 3
        // action space : Box(2)
        // broadcast std
        $mean = $g->Variable($la->array([[3,4],[5,6],[7,8]]));  // (batchSize,numActions)
        $std  = $g->Variable($la->array([3,2]));                // (numActions)
        $dist = new Normal($nn,$mean,$std);
        $action = $dist->sample();
        $this->assertEquals([3,2],$action->shape());            // (batchSize,numActions)
        $this->assertEquals(NDArray::float32,$action->dtype());
        $this->assertInstanceof(Variable::class,$action);
        $action2 = $dist->sample();
        //var_dump($action);
        $this->assertFalse($la->isclose(
            $g->ndarray($action),
            $g->ndarray($action2)
        ));

        // batchSize : bactchShape(3)
        // action space : Box(2)
        $mean = $g->Variable($la->array([4,3]));    // (numActions)
        $std  = $g->Variable($la->array([2,1]));    // (numActions)
        $dist = new Normal($nn,$mean,$std);
        $action = $dist->sample(batchShape:[3]);
        $this->assertEquals([3,2],$action->shape());
        $this->assertEquals(NDArray::float32,$action->dtype());
        $this->assertInstanceof(Variable::class,$action);
        $action2 = $dist->sample(batchShape:[3]);
        //var_dump($action);
        $this->assertFalse($la->isclose(
            $g->ndarray($action),
            $g->ndarray($action2)
        ));

    }

    public function testLogProbShape()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : None
        // action space : Box(2)
        $mean = $g->Variable($la->array([4,3]));            // (numActions)
        $std  = $g->Variable($la->array([2,1]));            // (numActions)
        $dist = new Normal($nn,$mean,$std);
        $action = $dist->sample();
        $logProb = $dist->logProb($action);
        $this->assertEquals([2],$logProb->shape());         // (numActions)
        $this->assertEquals(NDArray::float32,$logProb->dtype());
        $this->assertInstanceof(Variable::class,$logProb);
        $action2 = $dist->sample();
        $logProb2 = $dist->logProb($action2);
        //var_dump($action);
        $this->assertFalse($la->isclose(
            $g->ndarray($logProb),
            $g->ndarray($logProb2)
        ));

        // batchSize : 3
        // action space : Box(1)
        $mean = $g->Variable($la->array([[5],[4],[3]]));    // (batchSize,numActions)
        $std  = $g->Variable($la->array([[3],[2],[1]]));    // (batchSize,numActions)
        $dist = new Normal($nn,$mean,$std);
        $action = $dist->sample();
        $logProb = $dist->logProb($action);
        $this->assertEquals([3,1],$logProb->shape());         // (batchSize,numActions)
        $this->assertEquals(NDArray::float32,$logProb->dtype());
        $this->assertInstanceof(Variable::class,$logProb);
        $action2 = $dist->sample();
        $logProb2 = $dist->logProb($action2);
        //var_dump($action);
        $this->assertFalse($la->isclose(
            $g->ndarray($logProb),
            $g->ndarray($logProb2)
        ));


        // batchSize : 3
        // action space : Box(2)
        // broadcast std
        $mean = $g->Variable($la->array([[3,4],[5,6],[7,8]]));  // (batchSize,numActions)
        $std  = $g->Variable($la->array([3,2]));                // (numActions)
        $dist = new Normal($nn,$mean,$std);
        $action = $dist->sample();
        $logProb = $dist->logProb($action);
        $this->assertEquals([3,2],$logProb->shape());           // (batchSize,numActions)
        $this->assertEquals(NDArray::float32,$logProb->dtype());
        $this->assertInstanceof(Variable::class,$logProb);
        $action2 = $dist->sample();
        $logProb2 = $dist->logProb($action2);
        //var_dump($action);
        $this->assertFalse($la->isclose(
            $g->ndarray($logProb),
            $g->ndarray($logProb2)
        ));
    }

    public function testEntropyShape()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : None
        // action space : Box(2)
        $mean = $g->Variable($la->array([4,3]));    // (numActions)
        $std  = $g->Variable($la->array([2,1]));    // (numActions)
        $dist = new Normal($nn,$mean,$std);
        $entropy = $dist->entropy();
        $this->assertEquals([2],$entropy->shape());
        $this->assertEquals(NDArray::float32,$entropy->dtype());
        $this->assertInstanceof(Variable::class,$entropy);

        // batchSize : 3
        // action space : Box(1)
        $mean = $g->Variable($la->array([[5],[4],[3]]));    // (batchSize,numActions)
        $std  = $g->Variable($la->array([[3],[2],[1]]));    // (batchSize,numActions)
        $dist = new Normal($nn,$mean,$std);
        $entropy = $dist->entropy();
        $this->assertEquals([3,1],$entropy->shape());        // (batchSize,numActions)
        $this->assertEquals(NDArray::float32,$entropy->dtype());
        $this->assertInstanceof(Variable::class,$entropy);

        // batchSize : 3
        // action space : Box(2)
        // broadcast std
        $mean = $g->Variable($la->array([[3,4],[5,6],[7,8]]));  // (batchSize,numActions)
        $std  = $g->Variable($la->array([3,2]));                // (numActions)
        $dist = new Normal($nn,$mean,$std);
        $entropy = $dist->entropy();
        $this->assertEquals([3,2],$entropy->shape());            // (batchSize,numActions)
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
        // action space : Box(2)
        // broadcast std
        $mean = $g->Variable($la->array([[3,4],[5,6],[7,8]]));  // (batchSize,numActions)
        $std  = $g->Variable($la->array([1e-7,1]));                // (numActions)
        $target = $g->Variable($la->array([[3,4],[5,6],[7,8]]));
        $lossfunc = $nn->losses()->MeanSquaredError();

        [$action,$loss] = $nn->with($tape=$g->GradientTape(),function () use ($nn,$mean,$std,$target,$lossfunc) {
            $dist = new Normal($nn,$mean,$std);
            $action = $dist->sample();
            $loss = $lossfunc($target,$action);
            return [$action,$loss];
        });
        [$dMean,$dStd] = $tape->gradient($loss, [$mean,$std]);
        //echo "action=".$la->toString($action)."\n";
        //echo "loss=".$la->toString($loss)."\n";
        //echo "dMean=".$la->toString($dMean)."\n";
        //echo "dStd=".$la->toString($dStd)."\n";
        //echo "ADD=".$la->toString($g->sub($action,$dMean))."\n";
        $this->assertTrue(true);
    }

    public function testlogProbCompute()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : 3
        // action space : Box(2)
        // broadcast std
        $mean = $g->Variable($la->array([[3,4],[5,6],[7,8]]));  // (batchSize,numActions)
        $std  = $g->Variable($la->array([1e-2,1]));                // (numActions)
        $target = $g->Variable($la->array([[3.0001,4.0001],[5.0001,6.0001],[7.0001,8.0001]]));
        $lossfunc = $nn->losses()->MeanSquaredError();

        [$logProb,$loss] = $nn->with($tape=$g->GradientTape(),function () use ($nn,$g,$mean,$std,$target,$lossfunc) {
            $dist = new Normal($nn,$mean,$std);
            $logProb = $dist->logProb($target);
            $loss = $lossfunc($g->zerosLike($logProb),$logProb);
            return [$logProb,$loss];
        });
        [$dMean,$dStd] = $tape->gradient($loss, [$mean,$std]);
        //echo "logProb=".$la->toString($logProb)."\n";
        //echo "loss=".$la->toString($loss)."\n";
        //echo "dMean=".$la->toString($dMean)."\n";
        //echo "dStd=".$la->toString($dStd)."\n";
        $trueslogProb = $la->array([
            [ 3.6861815, -0.9189385],
            [ 3.6861815, -0.9189385],
            [ 3.6861815, -0.9189385],
        ]);
        $this->assertTrue($la->isclose($trueslogProb,$logProb));
        $truesdMean = $la->array([
            [ 1.2299272e+00, -3.0672883e-05],
            [ 1.2299272e+00, -3.0672883e-05],
            [ 1.2299272e+00, -3.0672883e-05],
        ]);
        $this->assertTrue($la->isclose($truesdMean,$dMean,atol:1e-2));
        $truesdStd = $la->array([-368.58118, 0.9189386]);
        $this->assertTrue($la->isclose($truesdStd,$dStd));
    }

    public function testEntropyCompute()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $nn = $this->newBuilder($mo);
        $g = $nn->gradient();

        // batchSize : 3
        // action space : Box(2)
        // broadcast std
        $mean = $g->Variable($la->array([[3,4],[5,6],[7,8]]));  // (batchSize,numActions)
        $std  = $g->Variable($la->array([1e-2,1]));                // (numActions)
        $lossfunc = $nn->losses()->MeanSquaredError();

        [$entropy,$loss] = $nn->with($tape=$g->GradientTape(),function () use ($nn,$g,$mean,$std,$lossfunc) {
            $dist = new Normal($nn,$mean,$std);
            $entropy = $dist->entropy();
            $loss = $lossfunc($g->zerosLike($entropy),$entropy);
            return [$entropy,$loss];
        });
        [$dStd] = $tape->gradient($loss, [$std]);
        //echo "entropy=".$la->toString($entropy)."\n";
        //echo "loss=".$la->toString($loss)."\n";
        //echo "dStd=".$la->toString($dStd)."\n";
        $truesentropy = $la->array([
            [-3.1862316,  1.4189385],
            [-3.1862316,  1.4189385],
            [-3.1862316,  1.4189385],
        ]);
        $this->assertTrue($la->isclose($truesentropy,$entropy));
        $truesdStd = $la->array([-318.6232, 1.4189385]);
        $this->assertTrue($la->isclose($truesdStd,$dStd));
    }

}
