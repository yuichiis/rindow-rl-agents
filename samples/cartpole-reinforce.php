<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;
use Rindow\RL\Agents\Runner\EpisodeRunner;
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
$activation = null;
$learningRate = 2e-3; # 1e-2;#
$useBaseline = null; # true;#false;
$useNormalize = true;#false;


$env = new CartPoleV1($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();
$evalEnv = new CartPoleV1($la);

$agent = new Reinforce(
    $la,
    gamma:$gamma,
    useBaseline:$useBaseline,
    useNormalize:$useNormalize,
    nn:$nn,
    stateShape:$stateShape,
    numActions:$numActions,
    fcLayers:$fcLayers,
    activation:$activation,
    //lossFn:$lossFn,
    optimizerOpts:['lr'=>$learningRate],mo:$mo,
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

$filename = __DIR__.'\\cartpole-reinforce';
if(!$agent->fileExists($filename)) {
    $driver = new EpisodeRunner($la,$env,$agent,$maxExperienceSize,evalEnv:$evalEnv);
    $driver->metrics()->format('reward','%5.1f');
    $driver->metrics()->format('valRewards','%5.1f');
    $history = $driver->train(
        numIterations:$numIterations,
        metrics:['reward','loss','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,verbose:1,
    );
    $iter = $la->array($history['iter']);
    $arts[] = $plt->plot($iter,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['loss'],200,500))[0];
    $arts[] = $plt->plot($iter,$la->array($history['valRewards']))[0];
    //$plt->plot($ep,$la->scal(200,$la->array($history['epsilon'])));
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    $plt->legend($arts,['reward','loss','valRewards']);
    $plt->show();
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    [$state,$info] = $env->reset();
    $env->render();
    $done=false;
    $truncated=false;
    $testReward = 0;
    $testSteps = 0;
    while(!($done || $truncated)) {
        $action = $agent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
$filename = $env->show(path:__DIR__.'\\cartpole-reinforce-trained.gif');
echo "filename: {$filename}\n";
