<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\A2C\A2C;

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

$numIterations = 500000;#200000;#300;#1000;#
$logInterval =   1000;  #10; #
$evalInterval = 20000; #10; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;#100000;
$batchSize = 256;#32;#
$gamma = 0.99;
$valueLossWeight = 0.5;
$entropyWeight = 0.0;#0.01;
$fcLayers = [64,64];# [32,32];#
$learningRate = 7e-4;#1e-3;#1e-5;#

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
$dqnAgent = new A2C(
    $la,
    nn:$nn,stateShape:$stateShape,numActions:$numActions,fcLayers:$fcLayers,
    batchSize:$batchSize,gamma:$gamma,
    valueLossWeight:$valueLossWeight,entropyWeight:$entropyWeight,
    optimizerOpts:['lr'=>$learningRate],mo:$mo,
);
$dqnAgent->summary();

$filename = __DIR__.'\\cartpole-a2c';
if(!$dqnAgent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$dqnAgent,$maxExperienceSize);
    $driver = new StepRunner($la,$env,$dqnAgent,$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:null,
        metrics:['steps','reward','loss','entropy','valSteps','valReward'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,verbose:1
    );
    echo "\n";
    $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
    //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,$la->scal(200/max($history['loss']),$la->array($history['loss'])))[0];
    //$arts[] = $plt->plot($ep,$la->array($history['valSteps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valReward']))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    #$plt->legend($arts,['steps','reward','epsilon','loss','valSteps','valReward']);
    $plt->legend($arts,['reward','loss','valReward']);
    //$plt->legend($arts,['steps','valSteps']);
    $plt->show();
    $dqnAgent->saveWeightsToFile($filename);
} else {
    $dqnAgent->loadWeightsFromFile($filename);
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
        $action = $dqnAgent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'\\cartpole-a2c-trained.gif');
echo "filename: {$filename}\n";
