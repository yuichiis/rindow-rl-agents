<?php
namespace RindowTest\RL\Agents\ReplayBuffer\ReplayBufferTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use LogicException;
use InvalidArgumentException;

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

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $memory = new ReplayBuffer($la,$maxSize=3);
        $this->assertCount(0,$memory);

        $memory->add(['a','b']);
        $this->assertCount(1,$memory);
        $this->assertEquals(['a','b'],$memory->last());
        $res = $memory->sample(1);
        $this->assertCount(1,$res);
        $this->assertEquals(['a','b'],$res[0]);

        $memory->add(['c','d']);
        $memory->add(['e','f']);
        $this->assertCount(3,$memory);
        $this->assertEquals(['e','f'],$memory->last());
        $res = $memory->sample(2);
        $this->assertCount(2,$res);

        $memory->add(['g','h']);
        $this->assertCount(3,$memory);
        $this->assertEquals(['g','h'],$memory->last());
        $res = $memory->sample(3);
        $this->assertCount(3,$res);
    }

    public function testLastNoData()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $memory = new ReplayBuffer($la,$maxSize=3);
        $this->assertCount(0,$memory);

        $this->expectException(LogicException::class);
        $this->expectExceptionMessage('No data');
        $memory->last();
    }

    public function testTooFewItems()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $memory = new ReplayBuffer($la,$maxSize=3);
        $memory->add(['a','b']);
        $this->assertCount(1,$memory);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Too few items are stored.');
        $res = $memory->sample(3);
    }
}