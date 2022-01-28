<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

use Rindow\RL\Gym\Pendulum\PendulumV1;
use Rindow\RL\Gym\Core\Rendering\RenderFactory;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$env = new PendulumV1($la);

echo "observationSpace.shape [".implode(',',$env->observationSpace()->shape())."]\n";
echo "actionSpace.shape [".implode(',',$env->actionSpace()->shape())."]\n";
echo "actionSpace.high ".$mo->toString($env->actionSpace()->high())."\n";
echo "actionSpace.low ".$mo->toString($env->actionSpace()->low())."\n";
echo "actionSpace.high.dtype ".$mo->dtypeToString(($env->actionSpace()->high()->dtype()))."\n";

exit();

$env->reset();
$env->render();
$env->show();

$env->reset();
$maxSteps = 10;
$done = false;
$step = 0;
while(!$done && $step<$maxSteps) {
    $action = $la->randomUniform([1],-2,2);
    [$observation,$reward,$done,$info] = $env->step($action);
    $env->render();
    $step++;
}
$env->show(null,$delay=10);
