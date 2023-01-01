<?php
namespace RindowTest\RL\Agents\Util\RandomTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Util\Random;
use LogicException;
use InvalidArgumentException;

class TestRandom
{
    use Random;

    protected $la;

    public function __construct($la)
    {
        $this->la = $la;
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

    public function testRandomCategorical()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $rand = new TestRandom($la);

        $probabilities = $la->array([
            [2.0,  2.0,  2.0 ],
            [9.0,  2.0,  1.0 ],
            [0.1,  0.8,  0.1 ],
        ]);
        $actions = $rand->randomCategorical($probabilities, $numSamples=10);

        $this->assertEquals([3,10],$actions->shape());
        $this->assertEquals(NDArray::uint32,$actions->dtype());
        $this->assertLessThan(3,$la->max($actions));
    }
}