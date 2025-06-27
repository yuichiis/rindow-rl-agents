<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV0;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Agents\Agent\Reinforce\Reinforce;

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

$numIterations = 300;#1000;#
$logInterval =   10; #50; #
$evalInterval =  10; #50; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;#100000;
$gamma = 0.9;#
$convLayers = null;
$convType = null;
$fcLayers = [32,32];# [10,10];#
$learningRate = 1e-2;#1e-5;#
$useBaseline = true;#false;

$env = new CartPoleV0($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

$agent = new Reinforce(
    $la,
    gamma:$gamma,
    useBaseline:$useBaseline,
    nn:$nn,
    stateShape:$stateShape,
    numActions:$numActions,
    fcLayers:$fcLayers,
    activation:$activation,
    //lossFn:$lossFn,
    optimizerOpts:['lr'=>$learningRate],mo:$mo,
);
$agent->summary();

$filename = __DIR__.'\\cartpole-reinforce';
if(!$agent->fileExists($filename)) {
    $driver = new EpisodeDriver($la,$env,$agent,$maxExperienceSize);
    $history = $driver->train(numIterations:$numIterations,
        metrics:['reward','val_reward'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,verbose:1);
    $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
    $plt->plot($ep,$la->array($history['reward']));
    $plt->plot($ep,$la->array($history['val_reward']));
    //$plt->plot($ep,$la->scal(200,$la->array($history['epsilon'])));
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    $plt->legend(['reward','val_reward']);
    $plt->show();
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    echo ".";
    [$state,$info] = $env->reset();
    $env->render();
    $done=false;
    $truncated=false;
    while(!($done || $truncated)) {
        $action = $agent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $env->render();
    }
}
echo "\n";
$env->show();

