<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ActionMask;
use Rindow\RL\Agents\Util\GradientClipping;

class UtilityTest extends MatrixTestCase
{
    public function testActionMaskHasAny() : void
    {
        $this->assertFalse(ActionMask::hasAny(
            $this->la,$this->la->array([false,false],dtype:NDArray::bool)
        ));
        $this->assertTrue(ActionMask::hasAny(
            $this->la,$this->la->array([false,true],dtype:NDArray::bool)
        ));
    }

    public function testClipByGlobalNormScalesAllGradientsTogether() : void
    {
        $gradients = [$this->la->array([3.0,4.0]),$this->la->array([0.0,12.0])];
        $returned = GradientClipping::clipByGlobalNorm($this->la,$gradients,6.5);

        $this->assertSame($gradients,$returned);
        // global norm=13, therefore scale=0.5
        $this->assertArrayEqualsWithDelta([1.5,2.0],$this->arrayOf($gradients[0]));
        $this->assertArrayEqualsWithDelta([0.0,6.0],$this->arrayOf($gradients[1]));
    }

    public function testClipDoesNotChangeSmallGradients() : void
    {
        $gradient = $this->la->array([3.0,4.0]);
        GradientClipping::clipByGlobalNorm($this->la,[$gradient],10.0);
        $this->assertArrayEqualsWithDelta([3.0,4.0],$this->arrayOf($gradient));
    }
}
