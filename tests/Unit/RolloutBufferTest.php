<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Rindow\RL\Agents\ReplayBuffer\RolloutBuffer;

class RolloutBufferTest extends MatrixTestCase
{
    public function testGaeAndPolicyData() : void
    {
        $buffer = new RolloutBuffer($this->la,3,1,storePolicyData:true);
        $buffer->add($this->la->array([1]),0,1.0,false,false,0.5,-0.1);
        $buffer->add($this->la->array([2]),1,2.0,true,true,0.25,-0.2);

        [$obs,$actions,$logProb,$advantages,$returns,$values] =
            $buffer->finish(gamma:0.9,gaeLambda:0.8,lastValue:99.0);

        // t1: delta=2-0.25=1.75; t0: delta=1+0.9*0.25-0.5=0.725
        // A0=0.725+0.9*0.8*1.75=1.985
        $this->assertArrayEqualsWithDelta([1.985,1.75],$this->arrayOf($advantages));
        $this->assertArrayEqualsWithDelta([2.485,2.0],$this->arrayOf($returns));
        $this->assertArrayEqualsWithDelta([-0.1,-0.2],$this->arrayOf($logProb));
        $this->assertArrayEqualsWithDelta([0.5,0.25],$this->arrayOf($values));
        $this->assertArrayEqualsWithDelta([[1.0],[2.0]],$this->arrayOf($obs));
        $this->assertSame([0,1],$this->arrayOf($actions));
        $this->assertSame(0,$buffer->size());
    }

    public function testTruncationBootstrapsButStopsGaeAcrossEpisode() : void
    {
        $buffer = new RolloutBuffer($this->la,2,1);
        $buffer->add($this->la->array([0]),0,1.0,false,true,2.0);
        [,,$advantages,$returns] = $buffer->finish(0.5,1.0,4.0);

        $this->assertArrayEqualsWithDelta([1.0],$this->arrayOf($advantages));
        $this->assertArrayEqualsWithDelta([3.0],$this->arrayOf($returns));
    }

    public function testFullBufferRejectsAnotherTransition() : void
    {
        $buffer = new RolloutBuffer($this->la,1,1);
        $buffer->add($this->la->array([0]),0,0.0,false,false,0.0);
        $this->expectException(\OverflowException::class);
        $buffer->add($this->la->array([1]),0,0.0,false,false,0.0);
    }
}
