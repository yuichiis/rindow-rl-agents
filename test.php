<?php
require __DIR__.'/vendor/autoload.php';
use Interop\Polite\Math\Matrix\NDArray;


$mo = new Rindow\Math\Matrix\MatrixOperator();
$la = $mo->la();
function generateThresholds(NDArray $probs) : NDArray
{
    if($probs->ndim()!=2) {
        throw new InvalidArgumentException('probabilities must be 2D NDArray');
    }
    [$m,$n] = $probs->shape();
    $p2 = $la->zeros($la->alloc([$m,$n-1],$probs->dtype()));
    foreach ($probs as $key => $value) {
        $la->cumsum($value[[0,$n-2]],null,null,$p2[$key]);
    }
    return $p2;
}

$a = $la->array([
    [[-0.5,-1],[1,2]],
    [[-1,-1],[1,1]],
    [[0,0],[0,0]],
]);

echo count($a)."\n";
$shape = $a->shape();
$count = array_shift($shape);
$size = (int)array_product($shape);
$a = $a->reshape([$count,$size]);
var_dump($a->shape());
$b = $la->reduceSum($a,-1);
echo $mo->toString($b,"%3.3f")."\n";
$r = $la->reciprocal($la->copy($b));
echo $mo->toString($b,"%3.3f")."\n";
foreach ($a as $key => $value) {
    $bb = $b[$key];
    if($bb==0) {
        $la->fill(1/$size,$value);
    } else {
        $la->multiply($r[[$key,$key]],$la->expandDims($value,$axis=-1));
    }
}
echo $mo->toString($a,"%3.3f")."\n";
echo $mo->toString($la->nan2num($a,1),"%3.3f")."\n";
