<?php
namespace RindowTest\RL\Agents\ReplayBuffer\QueueBufferTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Agents\ReplayBuffer\QueueBuffer;
use LogicException;
use InvalidArgumentException;

class QueueBufferTest extends TestCase
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
        $memory = new QueueBuffer($la,$maxSize=3);
        $this->assertCount(0,$memory);

        $memory->add(['a','b']);
        $this->assertCount(1,$memory);
        $this->assertEquals(['a','b'],$memory->last());
        $res = $memory->sample(1);
        $this->assertCount(2,$res);
        $this->assertCount(1,$res[0]);
        $this->assertEquals(['a'],$res[0]);
        $this->assertCount(1,$res[1]);
        $this->assertEquals(['b'],$res[1]);

        $memory->add(['c','d']);
        $memory->add(['e','f']);
        $this->assertCount(3,$memory);
        $this->assertEquals(['e','f'],$memory->last());
        $res = $memory->sample(2);
        $this->assertCount(2,$res);
        $this->assertCount(2,$res[0]);
        $this->assertCount(2,$res[1]);

        $memory->add(['g','h']);
        $this->assertCount(3,$memory);
        $this->assertEquals(['g','h'],$memory->last());
        $res = $memory->sample(3);
        $this->assertCount(2,$res);
        $this->assertCount(3,$res[0]);
        $this->assertCount(3,$res[1]);
        $this->assertEquals([0,1,2],array_keys($res[0]));
        $this->assertEquals([0,1,2],array_keys($res[1]));
        $res = $memory->recently(3);
        $this->assertCount(2,$res);
        $this->assertCount(3,$res[0]);
        $this->assertCount(3,$res[1]);
        $this->assertEquals(['c','e','g'],$res[0]);
        $this->assertEquals(['d','f','h'],$res[1]);
        $this->assertEquals([0,1,2],array_keys($res[0]));
        $this->assertEquals([0,1,2],array_keys($res[1]));
        $res = $memory->get(-3);
        $this->assertCount(2,$res);
        $this->assertEquals(['c','d'],$res);
        $res = $memory->get(-2);
        $this->assertCount(2,$res);
        $this->assertEquals(['e','f'],$res);
        $res = $memory->get(-1);
        $this->assertCount(2,$res);
        $this->assertEquals(['g','h'],$res);

        $memory->clear();
        $this->assertCount(0,$memory);
    }

    public function testLastNoData()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $memory = new QueueBuffer($la,$maxSize=3);
        $this->assertCount(0,$memory);

        $this->expectException(LogicException::class);
        $this->expectExceptionMessage('No data');
        $memory->last();
    }

    public function testTooFewItems()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $memory = new QueueBuffer($la,$maxSize=3);
        $memory->add(['a','b']);
        $this->assertCount(1,$memory);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Too few items are stored.');
        $res = $memory->sample(3);
    }
}