<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Agents\Driver\StepDriver;
use Rindow\RL\Agents\Agent\Ddpg\Ddpg;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;
use Rindow\RL\Gym\Core\Rendering\RenderFactory;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);


$numIterations = 100;
$logInterval =   1; #
$evalInterval =  1; #10; #
$numEvalEpisodes = 0;
$maxExperienceSize = 50000;
$batchSize = 128;
$gamma = 0.99;
$stdDev = 0.2;
$noiseDecay = 0.999;
$minStdDev = 0.01;
$episodeAnnealing = true;

$actorNetworkOptions = [
    'fcLayers' => [256, 256],
];
$criticNetworkOptions = [
    'staFcLayers' => [256],
    'actLayers' => [],
    'comLayers' => [256]
];
# $staFcLayers = [16, 32];
# $actFcLayers = [32];
# $conFcLayers = [256,256];
$targetUpdatePeriod = 1;    #1;    #5;    #200;#5;    #5;   #5;   #200;#
$targetUpdateTau =    1e-3;#0.005;#0.005;#0.005;#1.0;#0.025;#0.01;#0.05;#1.0;#
$actorOptimizerOpts = [
    'lr'=>1e-4,#0.001;#0.001;
];
$criticOptimizerOpts = [
    'lr'=>1e-3,#0.002;#0.002;
    'epsilon'=>1e-2,
];


$env = new PendulumV1($la);

echo "observationSpace.shape (".implode(',',$env->observationSpace()->shape()).")\n";
echo "actionSpace.shape (".implode(',',$env->actionSpace()->shape()).")\n";
echo "actionSpace.high ".$mo->toString($env->actionSpace()->high())."\n";
echo "actionSpace.low ".$mo->toString($env->actionSpace()->low())."\n";
echo "actionSpace.high.dtype ".$mo->dtypeToString(($env->actionSpace()->high()->dtype()))."\n";

//exit();

$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->shape()[0]; # DDPG handles continuous actions
$lowerBound = $env->actionSpace()->low();
$upperBound = $env->actionSpace()->high();

$ddpgAgent = new Ddpg($la,$nn,
    $stateShape,$numActions,$lowerBound,$upperBound,
    stdDev:$stdDev,
    batchSize:$batchSize,
    gamma:$gamma,
    targetUpdatePeriod:$targetUpdatePeriod,
    targetUpdateTau:$targetUpdateTau,
    criticOptimizerOpts:$criticOptimizerOpts,
    actorOptimizerOpts:$actorOptimizerOpts,
    noiseDecay:$noiseDecay,
    minStdDev:$minStdDev,
    actorNetworkOptions:$actorNetworkOptions,
    criticNetworkOptions:$criticNetworkOptions,
    episodeAnnealing:$episodeAnnealing,
);
$ddpgAgent->summary();

$env->reset();
//$env->render();
//$env->show();

$filename = __DIR__.'\\pendulum-ddpg';
if(!$ddpgAgent->fileExists($filename)) {
    $driver = new EpisodeDriver($la,$env,$ddpgAgent,$maxExperienceSize);
    //$driver = new StepDriver($la,$env,$ddpgAgent,$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:$maxSteps=null,
        metrics:$metrics=['steps','reward','epsilon','loss','val_steps','val_reward'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,logInterval:$logInterval,
        verbose:$verbose=2);
    //echo "\n";
    if($evalInterval>0) {
        $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
        //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
        $rewards = $la->array($history['reward']);
        $min = $la->min($rewards);
        $losses = $la->increment($la->scal(-$min/max($history['loss']),$la->array($history['loss'])),$min);
        $arts[] = $plt->plot($ep,$rewards)[0];
        $arts[] = $plt->plot($ep,$losses)[0];
        //$arts[] = $plt->plot($ep,$la->scal(-$min,$la->array($history['epsilon'])))[0];
        if($numEvalEpisodes > 0) {
            //$arts[] = $plt->plot($ep,$la->array($history['val_steps']))[0];
            $arts[] = $plt->plot($ep,$la->array($history['val_reward']))[0];
        }
        $plt->xlabel('Iterations');
        $plt->ylabel('Reward');
        #$legends = ['Policy Gradient','Sarsa'];
        #$legends = ['steps','reward','loss','epsilon'];
        $legends = ['reward','loss'];
        #$legends = ['steps','val_steps'];
        if($numEvalEpisodes > 0) {
            #$legends[] = 'val_steps';
            $legends[] = 'val_reward';
        }
        $plt->legend($arts,$legends);
        $plt->show();
    }
    $ddpgAgent->saveWeightsToFile($filename);
} else {
    $ddpgAgent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    echo ".";
    [$state,$info] = $env->reset();
    $env->render();
    $maxSteps = 200;
    $done=false;
    $truncated=false;
    $step = 0;
    while(!($done||$truncated) && $step<$maxSteps) {
        $action = $ddpgAgent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $env->render();
        $step++;
    }
}
echo "\n";
$env->show();

//$env->reset();
//$maxSteps = 10;
//$done = false;
//$step = 0;
//while(!$done && $step<$maxSteps) {
//    $action = $la->randomUniform([1],-2,2);
//    [$state,$reward,$done,$truncated,$info] = $env->step($action);
//    $env->render();
//    $step++;
//}
//$env->show(null,$delay=10);
