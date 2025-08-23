<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\PPO\PPO;

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


$numIterations = 300000;# 30000;# 300;# 1000;#
$targetScore = null; # 475;
$numAchievements = null; # 5;
$logInterval =   null; # 1000;  # 10; #
$evalInterval =  2048; # 10; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;# 100000;
$rolloutSteps = 2048;
$batchSize = 64;#32;#
$epochs = 10;
$gamma = 0.99;
$gaeLambda = 0.95;
$valueLossWeight = 0.5;
$entropyWeight = 0.001;
$fcLayers = [128,128]; # [64,64];  # 
$learningRate = 3e-4; #1e-3;#1e-5;#
$normAdv = true;
$clipValueLoss = true;
$clipnorm = 0.5;

$env = new MountainCarV0($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

//$env->reset();
//$env->render();
//$env->show();
//exit();

$evalEnv = new MountainCarV0($la);

$customReward = function($env,$stepCount,$state,$action,$nextState,$reward,$done,$truncated,$info) {
    $position = $state[0];
    $velocity = $state[1];
    $nextPosition = $nextState[0];
    $nextVelocity = $nextState[1];
    $gravity = 9.81;

    $mass = 1.0; #正規化定数;
    $maxHeight = 2.0;
    $maxVelocity = 0.07;

    # ---- 位置エネルギー -----------------------------------------
    # MountainCar のトラック高さ: h(p) = sin(3p) + 1.0
    $height = sin(3 * $position) + 1.0;
    $potential = $mass * $gravity * $height;

    # ---- 運動エネルギー ---------------------------------------
    $kinetic = 0.5 * $mass * $velocity ** 2;
    
    # ---- 総エネルギー ---------------------------------------
    $totalEnergy = $potential + $kinetic;

    # ---- MAXエネルギー ---------------------------------------
    $maxPotential = $mass * $gravity * $maxHeight;
    $maxKinetic = 0.5 * $mass * $maxVelocity ** 2;
    $maxEnergy = $maxPotential + $maxKinetic;

    # ---- 報酬 ---------------------------------------
    $reward = ($totalEnergy - $maxEnergy)/$maxEnergy;

    if(($done || $truncated) && $nextPosition>=0.5) {
        $reward += 100;
    }
    return $reward;
};

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
    if ($done && $nextPosition >= 0.5) {
        $goalBonus = 100;
    }

    // ---- 最終的な報酬の計算 ----
    // 3つの要素をすべて合計する
    $finalReward = $energyReward + $stepPenalty + $goalBonus;
    //echo "finalReward: {$finalReward}, energyReward: {$energyReward}, stepPenalty: {$stepPenalty}, goalBonus: {$goalBonus}\n";
    
    return $finalReward;
};


//$pos = -0.5*pi()/3;
//echo "sin(pos)=".sin($pos*3)." cos(pos)=".cos($pos*3)."\n";
//$reward = $customReward($env,1,$la->array([$pos,0]),null,$la->array([-0.5,0]),null,null,null,null);
//echo "reward = {$reward}\n";
//exit();

//$network = new QNetwork($la,$nn,$stateShape,$numActions,$convLayers,$convType,$fcLayers);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$epsDecayRate);
$agent = new PPO(
    $la,
    nn:$nn,stateShape:$stateShape,numActions:$numActions,
    rolloutSteps:$rolloutSteps,epochs:$epochs,batchSize:$batchSize,
    fcLayers:$fcLayers,
    gamma:$gamma,gaeLambda:$gaeLambda,
    valueLossWeight:$valueLossWeight,entropyWeight:$entropyWeight,
    clipValueLoss:$clipValueLoss,normAdv:$normAdv,
    optimizerOpts:['lr'=>$learningRate],clipnorm:$clipnorm,mo:$mo,
);
$agent->summary();
$agent->setCustomRewardFunction($customGainReward);

function fitplot($la,array $x,float $window,float $bottom) : NDArray
{
    $scale = $window/(max($x)-min($x));
    $bias = -min($x)*$scale+$bottom;
    return $la->increment($la->scal($scale,$la->array($x)),$bias);
}

$filename = __DIR__.'\\mountaincar-ppo';
if(!$agent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$agent,$maxExperienceSize);
    $driver = new StepRunner($la,$env,$agent,$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $driver->metrics()->format('steps', '%5.1f');
    $driver->metrics()->format('reward','%5.1f');
    $driver->metrics()->format('Ploss','%+5.2e');
    $driver->metrics()->format('Vloss','%+5.2e');
    $driver->metrics()->format('valSteps', '%5.1f');
    $driver->metrics()->format('valRewards', '%5.1f');
    $history = $driver->train(
        numIterations:$numIterations,
        metrics:['steps','reward','Ploss','Vloss','entropy','valSteps','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,verbose:1,
    );
    $ep = $la->array($history['iter']);
    $arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['Ploss'],100,100))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['Vloss'],100,100))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['entropy'],100,100))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valSteps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valRewards']))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    $plt->legend($arts,['steps','reward','Ploss','Vloss','entropy','valSteps','valRewards']);
    //$plt->legend($arts,['reward','Ploss','Vloss','entropy','valRewards']);
    //$plt->legend($arts,['steps','valSteps']);
    $plt->show();
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    [$state,$info] = $agent->reset($env);
    $env->render();
    $done=false;
    $truncated=false;
    $testReward = 0;
    $testSteps = 0;
    while(!($done||$truncated)) {
        [$state,$reward,$done,$truncated,$info] = $agent->step($env,$testSteps,$state,info:$info);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'\\mountaincar-ppo-trained.gif');
echo "filename: {$filename}\n";
