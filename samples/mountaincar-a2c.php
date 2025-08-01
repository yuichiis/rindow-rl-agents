<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\A2C\A2C;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);
$work = new ArrayObject();

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

$customReward = function($env,$stepCount,$state,$action,$nextState,$reward,$done,$truncated,$info) use ($work) {
    $nextPosition = $nextState[0];
    $nextVelocity = $nextState[1];
    $position = $state[0];
    $velocity = $state[1];
    $gravity = 0.0025;

    $c = 1 / ($gravity*sin(3*0.5) + 0.5*0.07*0.07); #正規化定数

    $nextEnergy = $c*($gravity*sin(3*$nextPosition) + 0.5*$nextVelocity*$nextVelocity);
    $energy = $c*($gravity*sin(3*$position) + 0.5*$velocity*$velocity);
    $energyGain = $nextEnergy - $energy;
    $reward += $energyGain*10;
    if(($done || $truncated) && $nextPosition>=0.5) {
        $reward += 100;
    }
    return $reward;
};

$numIterations = 150000;#300;#1000;#
$logInterval =   1000;  #10; #
$evalInterval = 20000; #10; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;#100000;
$batchSize = 256;#32;#
$gamma = 0.99;
$valueLossWeight = 0.5;
$entropyWeight = 0.01;
$convLayers = null;
$convType = null;
$fcLayers = [64,64];# [32,32];#
$learningRate = 1e-4;#1e-5;#

$env = new MountainCarV0($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

//$env->reset();
//$env->render();
//$env->show();
//exit();

$evalEnv = new MountainCarV0($la);
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

$filename = __DIR__.'\\mountaincar-a2c';
if(!$dqnAgent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$dqnAgent,$maxExperienceSize);
    $driver = new StepRunner($la,$env,$dqnAgent,$maxExperienceSize,evalEnv:$evalEnv);
    $driver->setCustomRewardFunction($customReward);
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:null,
        metrics:['steps','reward','loss','val_steps','val_reward'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,verbose:1,
    );
    echo "\n";
    $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
    //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    //$arts[] = $plt->plot($ep,$la->scal(200/max($history['loss']),$la->array($history['loss'])))[0];
    //$arts[] = $plt->plot($ep,$la->array($history['val_steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['val_reward']))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    #$plt->legend($arts,['steps','reward','epsilon','loss','val_steps','val_reward']);
    //$plt->legend($arts,['reward','loss','val_reward']);
    $plt->legend($arts,['reward','val_reward']);
    //$plt->legend($arts,['steps','val_steps']);
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
        [$nextState,$reward,$done,$truncated,$info] = $env->step($action);
        //$testReward += $reward;
        $testReward += $customReward($env,$testSteps,$state,$action,$nextState,$reward,$done,$truncated,$info);
        $testSteps++;
        $state = $nextState;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
$filename = $env->show(path:__DIR__.'\\mountaincar-a2c-trained.gif');
echo "filename: {$filename}\n";
