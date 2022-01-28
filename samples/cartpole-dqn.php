<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\CartPole\CartPoleV0;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Agents\Driver\StepDriver;
use Rindow\RL\Agents\Agent\Dqn;
use Rindow\RL\Agents\Network\QNetwork;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

## { $numIterations = 300; $evalInterval = 50; $numEvalEpisodes = 10;
##   $maxExperienceSize = 10000; $batchSize = 32;
##   $gamma = 0.99; $fcLayers = [100]; $targetUpdatePeriod = 5; $targetUpdateTau = 0.025;
##   $learningRate = 1e-3; $epsStart = 1.0; $epsStop = 0.05; $decayRate = 0.001;
##   $ddqn = false; $lossFn = null;}
## { $numIterations = 300; $evalInterval = 10; $numEvalEpisodes = 10;
##   $maxExperienceSize = 10000; $batchSize = 64;
##   $gamma = 0.99; $fcLayers = [100]; $targetUpdatePeriod = 5; $targetUpdateTau = 1.0;
##   $learningRate = 1e-3; $epsStart = 1.0; $epsStop = 0.05; $decayRate = 0.001;
##   $ddqn = true; $lossFn = $nn->losses->MeanSquaredError();}

$numIterations = 20000;#300;#1000;#
$logInterval =   200;  #10; #
$evalInterval =  1000; #10; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;#100000;
$batchSize = 64;#32;#
$gamma = 1.0;#0.99;#
$convLayers = null;
$convType = null;
$fcLayers = [100];# [32,32];#
$targetUpdatePeriod = 5;  #5;    #5;   #5;   #200;#
$targetUpdateTau =    1.0;#0.025;#0.01;#0.05;#1.0;#
$learningRate = 1e-3;#1e-5;#
$epsStart = 1.0; #1.0; #0.9;#1.0; #
$epsStop =  0.05;#0.01;#0.1;#0.05;#
$decayRate = 0.001;#0.0005;#
$ddqn = false;#true;#
$lossFn = $nn->losses->MeanSquaredError();

$env = new CartPoleV0($la);
$obsSize = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

//$env->reset();
//$env->render();
//$env->show();
//exit();

$evalEnv = new CartPoleV0($la);
$network = new QNetwork($la,$nn,$obsSize,$numActions,$convLayers,$convType,$fcLayers);
$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$decayRate);
$dqnAgent = new Dqn(
    $la,$network,$policy,
    $batchSize,$gamma,$targetUpdatePeriod,$targetUpdateTau,
    $ddqn,null,$lossFn,null,null,['lr'=>$learningRate],
    null,null,null,null,null,null,$mo
);
$dqnAgent->summary();
//$driver3 = new EpisodeDriver($la,$env,$dqnAgent,$maxExperienceSize);
$driver3 = new StepDriver($la,$env,$dqnAgent,$maxExperienceSize,null,null,$evalEnv);
$drivers = [$driver3];
$arts = [];
foreach ($drivers as $driver) {
    $driver->agent()->initialize();
    $history = $driver->train($numIterations,$maxSteps=null,
        $metrics=['steps','reward','epsilon','loss','val_steps','val_reward'],
        $evalInterval,$numEvalEpisodes,$logInterval,$verbose=1);
    echo "\n";
    $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
    //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,$la->scal(200/max($history['loss']),$la->array($history['loss'])))[0];
    //$arts[] = $plt->plot($ep,$la->array($history['val_steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['val_reward']))[0];
    $arts[] = $plt->plot($ep,$la->scal(200,$la->array($history['epsilon'])))[0];
}
$plt->xlabel('Iterations');
$plt->ylabel('Reward');
//$plt->legend($arts,['Policy Gradient','Sarsa']);
#$plt->legend($arts,['steps','reward','epsilon','loss','val_steps','val_reward']);
$plt->legend($arts,['reward','loss','val_reward','epsilon']);
//$plt->legend($arts,['steps','val_steps']);
$plt->show();


echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    echo ".";
    $observation = $env->reset();
    $env->render();
    $done=false;
    while(!$done) {
        $action = $dqnAgent->action($observation,$training=false);
        [$observation,$reward,$done,$info] = $env->step($action);
        $env->render();
    }
}
echo "\n";
$env->show();
