<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\Drivers\Selector;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

abstract class MatrixTestCase extends TestCase
{
    protected object $la;
    protected NeuralNetworks $nn;

    protected function setUp() : void
    {
        $service = (new Selector())->select();
        $mo = new MatrixOperator(service:$service);
        $this->la = $mo->la();
        $this->nn = new NeuralNetworks($mo);
    }

    protected function arrayOf(object $array) : array
    {
        return $this->la->toNDArray($array)->toArray();
    }

    protected function assertArrayEqualsWithDelta(
        array $expected,
        array $actual,
        float $delta=1.0e-6,
    ) : void {
        $this->assertCount(count($expected),$actual);
        foreach ($expected as $i => $value) {
            if (is_array($value)) {
                $this->assertArrayEqualsWithDelta($value,$actual[$i],$delta);
            } else {
                $this->assertEqualsWithDelta($value,$actual[$i],$delta);
            }
        }
    }
}
