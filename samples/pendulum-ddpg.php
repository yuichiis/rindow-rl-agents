<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Agents\Driver\StepDriver;
use Rindow\RL\Agents\Agent\Ddpg;
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
$batchSize = 64;
$gamma = 0.99;
$std_dev = 0.2;
# $fcLayers = [256, 256];
# $obsFcLayers = [16, 32];
# $actFcLayers = [32];
# $conFcLayers = [256,256];
$targetUpdatePeriod = 1;    #1;    #5;    #200;#5;    #5;   #5;   #200;#
$targetUpdateTau =    0.005;#0.005;#0.005;#1.0;#0.025;#0.01;#0.05;#1.0;#
$critic_lr = 0.002;#0.002;
$actor_lr  = 0.001;#0.001;


$env = new PendulumV1($la);

echo "observationSpace.shape [".implode(',',$env->observationSpace()->shape())."]\n";
echo "actionSpace.shape [".implode(',',$env->actionSpace()->shape())."]\n";
echo "actionSpace.high ".$mo->toString($env->actionSpace()->high())."\n";
echo "actionSpace.low ".$mo->toString($env->actionSpace()->low())."\n";
echo "actionSpace.high.dtype ".$mo->dtypeToString(($env->actionSpace()->high()->dtype()))."\n";

//exit();

$obsSize = $env->observationSpace()->shape();
$actionSize = $env->actionSpace()->shape();
$lower_bound = $env->actionSpace()->low();
$upper_bound = $env->actionSpace()->high();

$ddpgAgent = new Ddpg($la,$nn,
    $obsSize,$actionSize,$lower_bound,$upper_bound,
    std_dev:$std_dev,
    batchSize:$batchSize,
    gamma:$gamma,
    targetUpdatePeriod:$targetUpdatePeriod,
    targetUpdateTau:$targetUpdateTau,
    criticOptimizerOpts:['lr'=>$critic_lr],
    actorOptimizerOpts:['lr'=>$actor_lr],
);
//$ddpgAgent->summary();

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
    $observation = $env->reset();
    $env->render();
    $maxSteps = 200;
    $done=false;
    $step = 0;
    while(!$done && $step<$maxSteps) {
        $action = $ddpgAgent->action($observation,$training=false);
        [$observation,$reward,$done,$info] = $env->step($action);
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
//    [$observation,$reward,$done,$info] = $env->step($action);
//    $env->render();
//    $step++;
//}
//$env->show(null,$delay=10);
