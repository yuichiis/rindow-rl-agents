<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\DDPG\DDPG;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;
use Rindow\RL\Gym\Core\Rendering\RenderFactory;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);


$numIterations = 50000;
$logInterval =  null; #
$evalInterval =  1000; #10; #
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
$targetUpdatePeriod = 1;    #1;    #5;    # 200;#5;    #5;   #5;   # 200;#
$targetUpdateTau =    1e-3;#0.005;#0.005;#0.005;#1.0;#0.025;#0.01;#0.05;#1.0;#
$actorOptimizerOpts = [
    'lr'=>1e-4,#0.001;#0.001;
];
$criticOptimizerOpts = [
    'lr'=>1e-3,#0.002;#0.002;
    'epsilon'=>1e-2,
];


$env = new PendulumV1($la);
$evalEnv = new PendulumV1($la);

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

$agent = new DDPG($la,$nn,
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
$agent->summary();

$env->reset();
//$env->render();
//$env->show();

function fitplot(object $la,array $x,float $window,float $bottom) : NDArray
{
    $width = max($x)-min($x);
    if($width==0) {
        $scale = 1.0;
        $bias = $bottom;
    } else {
        $scale = $window/(max($x)-min($x));
        $bias = -min($x)*$scale+$bottom;
    }
    return $la->increment($la->scal($scale,$la->array($x)),$bias);
}

$filename = __DIR__.'\\pendulum-ddpg';
if(!$agent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$agent,$maxExperienceSize);
    $driver = new StepRunner($la,$env,$agent,experienceSize:$maxExperienceSize,evalEnv:$evalEnv);
    $driver->metrics()->format('steps','%5.1f');
    $driver->metrics()->format('reward','%7.1f');
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$numIterations,
        metrics:$metrics=['steps','reward','epsilon','loss','valSteps','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,logInterval:$logInterval,
        verbose:$verbose=1);
    //echo "\n";
    if($evalInterval>0) {
        $ep = $la->array($history['iter']);
        //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
        $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
        $arts[] = $plt->plot($ep,fitplot($la,$history['loss'],200,0))[0];
        //$rewards = $la->array($history['reward']);
        //$min = $la->min($rewards);
        //$losses = $la->increment($la->scal(-$min/max($history['loss']),$la->array($history['loss'])),$min);
        //$arts[] = $plt->plot($ep,$rewards)[0];
        //$arts[] = $plt->plot($ep,$losses)[0];
        //$arts[] = $plt->plot($ep,$la->scal(-$min,$la->array($history['epsilon'])))[0];
        if($numEvalEpisodes > 0) {
            //$arts[] = $plt->plot($ep,$la->array($history['val_steps']))[0];
            $arts[] = $plt->plot($ep,$la->array($history['valRewards']))[0];
        }
        $plt->xlabel('Iterations');
        $plt->ylabel('Reward');
        #$legends = ['Policy Gradient','Sarsa'];
        #$legends = ['steps','reward','loss','epsilon'];
        $legends = ['reward','loss'];
        #$legends = ['steps','val_steps'];
        if($numEvalEpisodes > 0) {
            #$legends[] = 'val_steps';
            $legends[] = 'valRewards';
        }
        $plt->legend($arts,$legends);
        $plt->show();
    }
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<1;$i++) {
    [$state,$info] = $env->reset();
    $env->render();
    $maxSteps = 210;
    $done=false;
    $truncated=false;
    $testReward = 0;
    $testSteps = 0;
    while(!($done||$truncated) && $testSteps<$maxSteps) {
        $action = $agent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'\\pendulum-ddpg-trained.gif');
echo "filename: {$filename}\n";
