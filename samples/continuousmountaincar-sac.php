<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\SAC\SAC;

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


$numIterations = 50000;# 100000; # 300; # 1000; #
$targetScore = null; # -250; #
$numAchievements = null; # 10; #
$logInterval =   10; #32; # 1000;  # 10; #
$evalInterval =  1024;# 256; # 10; #
$numEvalEpisodes = 10; # 100;
$maxExperienceSize = 50000; # 100000;
$rolloutSteps = 32;
$epochs = 32;
$batchSize = 512; # 100;# 32;#
$startSteps = 0; # 5000; # $batchSize;
$useSdeAtWarmup = true;
$logStdInit = -3.67;
$gamma = 0.9999;
//$gamma = 0.9;#  # <= continuousmountaincar 0.9;# 0.99;
$initailAlpha = 0.2;
$autoTuneAlpha = true; #true;
$targetEntropy = -1.0; // -numActions
$targetUpdatePeriod = 1;    #1;    #5;    # 200;#5;    #5;   #5;   # 200;#
$targetUpdateTau = 1e-2; #1e-3;#0.005;#0.005;#0.005;#1.0;#0.025;#0.01;#0.05;#1.0;#
$fcLayers = [64,64];
$learningRate = 3e-4;# 1e-3;#1e-5;#
//$actorOptimizerOpts = [
//    'lr' => 1e-4,
//];
//$criticOptimizerOpts = [
//    'lr' => $learningRate,
//];

// Initializing the action layer kernel for continuousmountaincar
//$std = 2.7; #2.7;
//$minval = log($std-0.003);
//$maxval = log($std+0.003);
//$minval = 0.05;
//$maxval = 0.05;
//$logStdInitializer = $nn->backend()->getInitializer('random_uniform',minval:$minval,maxval:$maxval);
//$actorNetworkOptions = [
//    'logStdKernelInitializer' => $logStdInitializer,
//    //'logStdBiasInitializer' => $logStdInitializer,
//];
$logStdInitializer = null;

$env = new ContinuousMountainCarV0($la);
$stateShape = $env->observationSpace()->shape();
$actionSpace = $env->actionSpace();

//$env->reset();
//$env->render();
//$env->show();
//exit();

$evalEnv = new ContinuousMountainCarV0($la);
/*
$customGainReward = function($env, $stepCount, $state, $action, $nextState, $reward, $done, $truncated, $info) {
    $position = $state[0];
    $velocity = $state[1];
    $nextPosition = $nextState[0];
    $nextVelocity = $nextState[1];

    // ---- 1. エネルギー増加による報酬（行動の指針）----
    $calculateEnergy = function($p, $v) {
        $potentialEnergy = sin(3 * $p); 
        $kineticEnergy = 0.5 * ($v ** 2);
        return $potentialEnergy + $kineticEnergy;
    };
    $energy = $calculateEnergy($position, $velocity);
    $nextEnergy = $calculateEnergy($nextPosition, $nextVelocity);
    
    // エネルギーの増加量に、学習を安定させるための係数をかける
    $energyReward = 10 * ($nextEnergy - $energy);

    // ---- 2. ステップペナルティ（効率化の促進）----
    // 毎ステップ、一定のコストを課す
    $stepPenalty = -0.1;

    // ---- 3. ゴールボーナス ----
    $goalBonus = 0;
    if ($done && $nextPosition >= 0.45) { // Continuousのゴールは0.5でなく0.45
        $goalBonus = 100;
    }

    // ---- 最終的な報酬の計算 ----
    // 3つの要素をすべて合計する
    $finalReward = $energyReward + $stepPenalty + $goalBonus;
    //echo "finalReward: {$finalReward}, energyReward: {$energyReward}, stepPenalty: {$stepPenalty}, goalBonus: {$goalBonus}\n";
    
    return $finalReward;
};
*/
//$customScaleReward = function($env, $stepCount, $state, $action, $nextState, $reward, $done, $truncated, $info) {
//    return $reward*10;
//};

//$network = new QNetwork($la,$nn,$stateShape,$numActions,$convLayers,$convType,$fcLayers);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$epsDecayRate);
$agent = new SAC(
    $la,
    nn:$nn,stateShape:$stateShape,actionSpace:$actionSpace,
    batchSize:$batchSize,
    rolloutSteps:$rolloutSteps,
    epochs:$epochs,
    startSteps:$startSteps,
    useSdeAtWarmup:$useSdeAtWarmup,
    fcLayers:$fcLayers,
    //initialStd:$initialStd,
    gamma:$gamma,
    initailAlpha:$initailAlpha,
    autoTuneAlpha:$autoTuneAlpha,
    targetEntropy:$targetEntropy,
    targetUpdatePeriod:$targetUpdatePeriod,
    targetUpdateTau:$targetUpdateTau,
    learningRate:$learningRate,
    //actorOptimizerOpts:$actorOptimizerOpts,
    //criticOptimizerOpts:$criticOptimizerOpts,
    logStdInit:$logStdInit,
    //actorNetworkOptions:$actorNetworkOptions,
    mo:$mo,
);
$agent->summary();
//$agent->setCustomRewardFunction($customGainReward);
//$agent->setCustomRewardFunction($customScaleReward);

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

$filename = __DIR__.'\\continuousmountaincar-sac';
if(!$agent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$agent,,experienceSize:$maxExperienceSize);
    $driver = new StepRunner($la,$env,$agent,experienceSize:$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $driver->metrics()->format('steps', '%5.1f');
    $driver->metrics()->format('reward','%5.1f');
    $driver->metrics()->format('Ploss','%+5.2e');
    $driver->metrics()->format('Vloss','%+5.2e');
    $driver->metrics()->format('Aloss','%+5.2e');
    $driver->metrics()->format('alpha','%+5.2e');
    $driver->metrics()->format('std','%6.4f');
    $driver->metrics()->format('valSteps', '%5.1f');
    $driver->metrics()->format('valRewards', '%5.1f');
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:null,
        //metrics:['steps','reward','Ploss','Vloss','Aloss','alpha','std','valSteps','valRewards'],
        metrics:['steps','reward','Ploss','Vloss','alpha','std','valSteps','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,targetScore:$targetScore,numAchievements:$numAchievements,verbose:1,
    );
    $iter = $la->array($history['iter']);
    $arts[] = $plt->plot($iter,fitplot($la,$history['steps'],100,200))[0];
    $arts[] = $plt->plot($iter,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['Ploss'],100,200))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['Vloss'],100,200))[0];
    //$arts[] = $plt->plot($iter,fitplot($la,$history['Aloss'],100,200))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['alpha'],100,200))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['std'],100,200))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['valSteps'],100,200))[0];
    $arts[] = $plt->plot($iter,$la->array($history['valRewards']))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['steps','reward','Ploss','Vloss','Aloss','alpha','valSteps','valRewards']);
    $plt->legend($arts,['steps','reward','Ploss','Vloss','alpha','std','valSteps','valRewards']);
    $plt->show();
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<1;$i++) {
    [$state,$info] = $agent->reset($env);
    $env->render();
    $done=false;
    $truncated=false;
    $testReward = 0;
    $testSteps = 0;
    while(!($done||$truncated)) {
        //$action = $agent->action($state,training:false,info:$info);
        //[$state,$reward,$done,$truncated,$info] = $env->step($action);
        [$state,$reward,$done,$truncated,$info] = $agent->step($env,$testSteps,$state,info:$info);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'\\continuousmountaincar-sac-trained.gif');
echo "filename: {$filename}\n";
