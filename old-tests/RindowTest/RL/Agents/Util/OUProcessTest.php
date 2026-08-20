<?php
namespace RindowTest\RL\Agents\Util\OUProcessTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Util\OUProcess;
use LogicException;
use InvalidArgumentException;

class OUProcessTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newLa($mo)
    {
        return $mo->la();
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
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $mean = $la->array([0]);
        $std  = $la->array([0.2]);
        $noise = new OUProcess($la,$mean,$std);
        $p1 = [];
        $dt = 1e-2;
        for($i=0;$i<20000;$i++) {
            $x1 = $noise->process();
            //[$x1,$x2] = $la->randomNormal($mean->shape(),0.0, sqrt($dt));
            $p1[] = $x1[0];
        }
        $p1 = $la->array($p1);
        $plt->plot($p1);
        $plt->legend(['p1']);
        $plt->title('OUProcess');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testGeneral()
    {
        $mo = $this->newMatrixOperator();
        $la = $this->newLa($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $mean = $la->array([1,   0]);
        $std  = $la->array([0.2, 0.5]);
        $noise = new OUProcess($la,$mean,$std);
        $p1 = [];
        $p2 = [];
        $dt = 1e-2;
        for($i=0;$i<100;$i++) {
            [$x1,$x2] = $noise->process();
            //[$x1,$x2] = $la->randomNormal($mean->shape(),0.0, sqrt($dt));
            $p1[] = $x1;
            $p2[] = $x2;
        }
        $p1 = $la->array($p1);
        $p2 = $la->array($p2);
        $plt->plot($p1);
        $plt->plot($p2);
        $plt->legend(['p1','p2']);
        $plt->title('OUProcess');
        $plt->show();
        $this->assertTrue(true);
    }
}