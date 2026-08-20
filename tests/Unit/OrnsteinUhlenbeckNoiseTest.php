<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Rindow\RL\Agents\Agent\DDPG\OrnsteinUhlenbeckNoise;

class OrnsteinUhlenbeckNoiseTest extends MatrixTestCase
{
    public function testZeroDiffusionStaysAtMeanAfterReset() : void
    {
        $noise = new OrnsteinUhlenbeckNoise(
            $this->la,3,sigma:0.0,theta:0.15,dt:0.01,mean:2.0
        );
        $this->assertArrayEqualsWithDelta([2.0,2.0,2.0],$this->arrayOf($noise->sample()));
        $noise->reset();
        $this->assertArrayEqualsWithDelta([2.0,2.0,2.0],$this->arrayOf($noise->sample()));
    }
}
