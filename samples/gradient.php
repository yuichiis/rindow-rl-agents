<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

class TestModel
{
    public function __construct(Type $var = null)
    {
        $this->call = $g->function([$this,'call']);
    }

    public function forward(...$args)
    {
        $call = $this->call;
        return $call(...$args);
    }

    public function call($a, $b)
    {
        return $this->g->add($a,$b);
    }

    public function doTrain(Type $var = null)
    {
        $loss = $nn->with($tape=$g->GradientTape(),
            function() {
                $call = $this->call;
                $predicts = $call(...$args);
                $this->lossfunc($trues,$predicts);
            }
        );
    }
}

function testfunc($a) {
    return $a;
}
//$testfunc = $g->function('testfunc');

$b = 9;
$a = fn($x)=>$b;
echo $a(123);