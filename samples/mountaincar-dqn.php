<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\DQN\DQN;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);
$work = new ArrayObject();

$customReward1 = function($env,$stepCount,$state,$action,$nextState,$reward,$done,$truncated,$info) use ($work) {
    $oid = spl_object_id($env);
    if(!isset($work[$oid])) {
        $work[$oid] = (object)['position'=>-INF,'velocity'=>-INF,'success'=>0];
    }
    [$position,$velocity] = $state;
    if($done&&$stepCount<199) {
        $work[$oid]->success++;
        //echo "success!! step=".$stepCount."\n";
    }
    if($work[$oid]->position < $position) {
        $work[$oid]->position = $position;
        //echo "max position=".$position."(".(abs(-0.5-$position)).")\n";
    }
    if($work[$oid]->velocity < $velocity) {
        $work[$oid]->velocity = $velocity;
        //echo "max velocity=".$velocity."\n";
    }

    //echo sprintf('pos=[%5.3f],vel=[%5.3f]',abs(-0.5-$position),abs($velocity))."\r";
    if(abs(-0.5-$position)<0.2 && abs($velocity)>0.015) {
        $reward += 1.0;
    }
    return $reward;
};

$customReward2 = function($env,$stepCount,$state,$action,$nextState,$reward,$done,$truncated,$info) use ($work) {
    $oid = spl_object_id($env);
    [$position,$velocity] = $state;
    //echo sprintf('pos=[%5.3f],vel=[%5.3f]',abs(-0.5-$position),abs($velocity))."\r";
    $reward = 0;
    if($done) {
        if($position>=0.5) {
            $work[$oid] = (isset($work[$oid]))?$work[$oid]+1:1;
            $reward = 1.0;
        } else {
            $reward = $position+0.5;
        }
    }
    return $reward;
};

$customReward3 = function($env,$stepCount,$state,$action,$nextState,$reward,$done,$truncated,$info) use ($work) {
    $oid = spl_object_id($env);
    if(!isset($work[$oid])) {
        $work[$oid] = (object)[
            'position_min'=>INF,'position_max'=>-INF,
            'velocity_min'=>INF,'velocity_max'=>-INF,
            'success'=>0,'steps'=>INF, 'velocity_cut'=>0
        ];
    }
    [$position,$velocity] = $state;
    if($done) {
        if($position>=0.5) {
            $work[$oid]->success++;
        } elseif($stepCount<199) {
            $work[$oid]->velocity_cut++;
        }
        if($work[$oid]->steps > $stepCount) {
            $work[$oid]->steps = $stepCount;
        }
    }
    if($work[$oid]->position_min > $position) {
        $work[$oid]->position_min = $position;
    }
    if($work[$oid]->position_max < $position) {
        $work[$oid]->position_max = $position;
    }
    if($work[$oid]->velocity_min > $velocity) {
        $work[$oid]->velocity_min = $velocity;
    }
    if($work[$oid]->velocity_max < $velocity) {
        $work[$oid]->velocity_max = $velocity;
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

// { $numIterations = 1000; $logInterval = 1; $evalInterval = 10; $numEvalEpisodes = 10;
//   $maxExperienceSize = 50000; $batchSize = 32; $gamma = 0.99; $convLayers = null; $convType = null;
//   $fcLayers = [16,16,16]; $activation = null; $targetUpdatePeriod = 10; $targetUpdateTau = 1e-2;
//   $learningRate = 1e-3; $epsStart = 0.1; $epsStop = 0.1; $decayRate = 0.0005; $ddqn = false;
//   $lossFn = $nn->losses->MeanSquaredError(); $episodeAnnealing = true;
//   // no custom reward
// }

$numIterations = 50000; # 1000; # 300;#
$logInterval =   100;  #5;  # 10; #
$evalInterval =  500; # 10; # 1000; #
$numEvalEpisodes = 10; # 10;
$maxExperienceSize = 50000;# 20000;# 100000;
$batchSize = 256; # 256;# 128;# 64;#
$gamma = 0.99;#1.0;#
$convLayers = null;
$convType = null;
$fcLayers = [128,128]; # [16,16,16]; # [128,52];# [24,48];# [128,64,32];# [32,32];#
$activation = null;#'tanh';
$targetUpdatePeriod = 10;  #-1; #5;    #5;   #5;   # 200;#
$targetUpdateTau =    1e-2;#1.0;#0.025;#0.01;#0.05;#1.0;#
$learningRate = 1e-3;#1e-5;#
$epsStart = 1.0;#1.0; #1.0; #0.9;#1.0; #
$epsStop =  0.1;#0.05;#0.01;#0.1;#0.05;#
$decayRate = 0.0005;#0.01;#0.001;#
$ddqn = true;#false;# 
$lossFn = null; #$nn->losses->MeanSquaredError();
$episodeAnnealing = false; #true;

$env = new MountainCarV0($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

//[$state,$info] = $env->reset();
//echo $mo->toString($state,'%1.2f')."\n";
//$env->render();
//$env->show();
//exit();

$evalEnv = new MountainCarV0($la);
$agent = new DQN(
    $la,
    nn:$nn,
    stateShape:$stateShape,numActions:$numActions,
    fcLayers:$fcLayers,
    batchSize:$batchSize,gamma:$gamma,
    epsStart:$epsStart,epsStop:$epsStop,epsDecayRate:$decayRate,episodeAnnealing:$episodeAnnealing,
    targetUpdatePeriod:$targetUpdatePeriod,targetUpdateTau:$targetUpdateTau,
    ddqn:$ddqn,lossFn:$lossFn,optimizerOpts:['lr'=>$learningRate],
    mo:$mo
);
$agent->summary();
$agent->setCustomRewardFunction($customGainReward);
//$driver = new EpisodeRunner($la,$env,$agent,$maxExperienceSize);
$driver = new StepRunner($la,$env,$agent,$maxExperienceSize,evalEnv:$evalEnv);

function fitplot($la,array $x,float $window,float $bottom) : NDArray
{
    $scale = $window/(max($x)-min($x));
    $bias = -min($x)*$scale+$bottom;
    return $la->increment($la->scal($scale,$la->array($x)),$bias);
}

$filename = __DIR__.'\\mountaincar-dqn';
if(!$agent->fileExists($filename)) {
    $driver->metrics()->format('steps', '%5.1f');
    $driver->metrics()->format('reward','%5.1f');
    $driver->metrics()->format('Ploss','%+5.2e');
    $driver->metrics()->format('Vloss','%+5.2e');
    $driver->metrics()->format('valSteps','%5.1f');
    $driver->metrics()->format('valRewards','%5.1f');
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:$maxSteps=null,
        metrics:['steps','reward','epsilon','loss','valSteps','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,verbose:$verbose=1
    );
    $ep = $la->array($history['iter']);
    $arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    //$arts[] = $plt->plot($ep,$la->scal(-1,$la->array($history['val_steps'])))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['loss'],100,100))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['epsilon'],100,100))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valSteps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valRewards']))[0];

    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    $plt->legend($arts,['steps','reward','loss','epsilon','valSteps','valRewards']);
    //$plt->legend($arts,['reward','loss','val_reward','epsilon']);
    //$plt->legend($arts,['steps','val_steps']);
    //$plt->legend($arts,['steps','reward']);
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
        [$state,$reward,$done,$truncated,$info] = $agent->step($env,$testSteps,$state,info:$info);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
$filename = $env->show(path:__DIR__.'\\mountaincar-dqn-trained.gif');
echo "filename: {$filename}\n";
