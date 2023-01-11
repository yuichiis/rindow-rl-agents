<?php
require __DIR__.'/vendor/autoload.php';
use Interop\Polite\Math\Matrix\NDArray;

$mo = new Rindow\Math\Matrix\MatrixOperator();
$la = $mo->la();
$a = $la->array([
    1,2,3,4
],NDArray::int32);

foreach($a as $i) {
    var_dump($i);
}