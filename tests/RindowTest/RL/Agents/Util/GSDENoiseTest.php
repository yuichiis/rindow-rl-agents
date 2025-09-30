<?php
namespace RindowTest\RL\Agents\Util\GSDENoiseTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Util\GSDENoise;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use LogicException;
use InvalidArgumentException;

class GSDENoiseTest extends TestCase
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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newBuilder($mo);
        $la = $nn->backend()->primaryLA();

        $g = $nn->gradient();
        //$plt = new Plot($this->getPlotConfig(),$mo);

        $actionDim = 2;
        $featureDim = 8;
        $logStdInit = -1;
        $noise = new GSDENoise($nn,$actionDim,$featureDim,$logStdInit);
        // sigma
        $sigma = $noise->sigma();
        $this->assertEquals([2],$sigma->shape());
        $sigma = $g->ndarray($sigma);
        $this->assertTrue($mo->la()->isclose(
            $mo->array([exp(-1),exp(-1)]),
            $sigma,
        ));

        // sample
        $mu = $g->Variable($la->randomNormal([2,$actionDim],0.0,1.0));
        $phi = $g->Variable($la->randomNormal([2,$featureDim],0.0,1.0));
        $muNDArray = $g->ndarray($mu);
        // mu != action
        $x0 = $noise->sample($mu,$phi);
        $x0 = $g->ndarray($x0);
        $this->assertFalse($mo->la()->isclose($x0,$muNDArray));
        // action == nextAction
        $x = $noise->sample($mu,$phi);
        $x = $g->ndarray($x);
        $this->assertTrue($mo->la()->isclose($x0,$x));
        // reset noise and then action != nextAction
        $noise->resetNoise();
        $x = $noise->sample($mu,$phi);
        $x = $g->ndarray($x);
        $this->assertFalse($mo->la()->isclose($x0,$x));
        // deterministic
        $x = $noise->sample($mu,$phi,deterministic:true);
        $x = $g->ndarray($x);
        $this->assertTrue($mo->la()->isclose($x,$muNDArray));
        // maybeResample
        $sdeSampleFreq = 8;
        $noise->maybeResample($sdeSampleFreq);
        $x0 = $noise->sample($mu,$phi);
        $x0 = $g->ndarray($x0);
        for($i=0;$i<$sdeSampleFreq-1;$i++) {
            $noise->maybeResample($sdeSampleFreq);
            $x = $noise->sample($mu,$phi);
            $x = $g->ndarray($x);
            $this->assertTrue($mo->la()->isclose($x0,$x));
        }
        $noise->maybeResample($sdeSampleFreq);
        $x = $noise->sample($mu,$phi);
        $x = $g->ndarray($x);
        $this->assertFalse($mo->la()->isclose($x0,$x));

    }

}