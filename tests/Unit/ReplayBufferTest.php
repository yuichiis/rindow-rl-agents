<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

class ReplayBufferTest extends MatrixTestCase
{
    public function testDiscreteTransitionAndActionMaskRoundTrip() : void
    {
        $buffer = new ReplayBuffer($this->la,1,2,actionMaskDimension:3);
        $buffer->add(
            $this->la->array([1.0,2.0],dtype:NDArray::float32),
            2,
            3.5,
            $this->la->array([4.0,5.0],dtype:NDArray::float32),
            true,
            $this->la->array([true,false,true],dtype:NDArray::bool),
        );

        [$obs,$actions,$rewards,$nextObs,$dones,$masks] = $buffer->sample(2);
        $this->assertSame([2,2],$obs->shape());
        $this->assertSame(NDArray::int32,$actions->dtype());
        $this->assertSame([2,3],$masks->shape());
        $this->assertArrayEqualsWithDelta([[1.0,2.0],[1.0,2.0]],$this->arrayOf($obs));
        $this->assertSame([2,2],$this->arrayOf($actions));
        $this->assertArrayEqualsWithDelta([3.5,3.5],$this->arrayOf($rewards));
        $this->assertArrayEqualsWithDelta([[4.0,5.0],[4.0,5.0]],$this->arrayOf($nextObs));
        $this->assertArrayEqualsWithDelta([1.0,1.0],$this->arrayOf($dones));
        $this->assertSame([[true,false,true],[true,false,true]],$this->arrayOf($masks));
    }

    public function testContinuousShapes() : void
    {
        $buffer = new ReplayBuffer($this->la,1,[2],actionDimension:2);
        $buffer->add(
            $this->la->array([1,2]),
            $this->la->array([0.25,-0.5]),
            1.0,
            $this->la->array([2,3]),
            false,
        );
        $batch = $buffer->sample(3);
        $this->assertCount(5,$batch);
        $this->assertSame([3,2],$batch[1]->shape());
        $this->assertSame([3,1],$batch[2]->shape());
        $this->assertSame([3,1],$batch[4]->shape());
    }

    public function testEmptyBufferCannotBeSampled() : void
    {
        $this->expectException(\UnderflowException::class);
        (new ReplayBuffer($this->la,2,1))->sample(1);
    }

    public function testMaskIsRequiredWhenConfigured() : void
    {
        $buffer = new ReplayBuffer($this->la,1,1,actionMaskDimension:2);
        $this->expectException(\InvalidArgumentException::class);
        $buffer->add($this->la->array([0]),0,0.0,$this->la->array([1]),false);
    }
}
