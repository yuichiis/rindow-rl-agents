<?php
namespace RindowTest\RL\Agents\Distribution\TanhBijectorTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Distribution\TanhBijector;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TanhBijectorTest extends TestCase
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

    public function testforwardNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $tanh = new TanhBijector($nn);
        $x = $g->Variable(0.5);
        $z = $tanh->forward($x);
        $this->assertTrue($mo->la()->isclose(
            $mo->array(0.46211715726),
            $z,
        ));
    }

    public function testAtanhNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $tanh = new TanhBijector($nn);
        $x = $g->Variable(0.5);
        $z = $tanh->atanh($x);
        $z = $K->ndarray($z);
        $this->assertTrue($mo->la()->isclose(
            $mo->array(0.5493061443340549),
            $z,
        ));
    }

    public function testinverseNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $tanh = new TanhBijector($nn);
        $x = $g->Variable(0.5);
        $z = $tanh->inverse($x);
        $z = $K->ndarray($z);
        $this->assertTrue($mo->la()->isclose(
            $mo->array(0.5493061443340549),
            $z,
        ));

        $tanh = new TanhBijector($nn);
        $x = $g->Variable(1.1);
        $z = $tanh->inverse($x);
        $z = $K->ndarray($z);
        $this->assertTrue($mo->la()->isclose(
            $mo->array(8.31776613691702),
            $z,
        ));
    }

    public function testlogProbCorrectionNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $tanh = new TanhBijector($nn);
        $x = $g->Variable(0.5);
        $z = $tanh->log_prob_correction($x);
        $z = $K->ndarray($z);
        $this->assertTrue($mo->la()->isclose(
            $mo->array(-0.24022775888443),
            $z,
        ));
    }

}