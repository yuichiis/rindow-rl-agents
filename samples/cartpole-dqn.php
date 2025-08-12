<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\DQN\DQN;
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
$targetUpdatePeriod = 5;  #5;    #5;   #5;   # 200;#
$targetUpdateTau =    1.0;#0.025;#0.01;#0.05;#1.0;#
$learningRate = 1e-3;#1e-5;#
$epsStart = 1.0; #1.0; #0.9;#1.0; #
$epsStop =  0.05;#0.01;#0.1;#0.05;#
$epsDecayRate = 0.001;#0.0005;#
$ddqn = true;#false;#
$lossFn = $nn->losses->MeanSquaredError();

$env = new CartPoleV1($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

//$env->reset();
//$env->render();
//$env->show();
//exit();

$evalEnv = new CartPoleV1($la);
//$network = new QNetwork($la,$nn,$stateShape,$numActions,$convLayers,$convType,$fcLayers);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$epsDecayRate);
$agent = new DQN(
    $la,
    nn:$nn,stateShape:$stateShape,numActions:$numActions,fcLayers:$fcLayers,
    epsStart:$epsStart,epsStop:$epsStop,epsDecayRate:$epsDecayRate,
    batchSize:$batchSize,gamma:$gamma,
    targetUpdatePeriod:$targetUpdatePeriod,targetUpdateTau:$targetUpdateTau,
    ddqn:$ddqn,lossFn:$lossFn,optimizerOpts:['lr'=>$learningRate],mo:$mo,
);
$agent->summary();

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

$filename = __DIR__.'\\cartpole-dqn';
if(!$agent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$agent,$maxExperienceSize);
    $driver = new StepRunner($la,env:$env,agent:$agent,
        experienceSize:$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(numIterations:$numIterations,
        metrics:['steps','reward','epsilon','loss','valSteps','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,verbose:1);
    echo "\n";
    $ep = $la->array($history['iter']);
    //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['loss'],200,0))[0];
    //$arts[] = $plt->plot($ep,$la->array($history['valSteps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valRewards']))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['epsilon'],200,0))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    #$plt->legend($arts,['steps','reward','epsilon','loss','valSteps','valRewards']);
    $plt->legend($arts,['reward','loss','valRewards','epsilon']);
    //$plt->legend($arts,['steps','val_steps']);
    $plt->show();
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<1;$i++) {
    [$state,$info] = $env->reset();
    $env->render();
    $done=false;
    $truncated=false;
    $testReward = 0;
    $testSteps = 0;
    while(!($done||$truncated)) {
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
$filename = $env->show(path:__DIR__.'\\cartpole-dqn-trained.gif');
echo "filename: {$filename}\n";
