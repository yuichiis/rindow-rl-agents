<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\DQN;
use Rindow\RL\Agents\Network\QNetwork;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);
$work = new ArrayObject();

$customReward2 = function($env,$stepCount,$state,$reward,$done,$info) use ($work) {
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

$customReward2 = function($env,$stepCount,$state,$reward,$done,$info) use ($work) {
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

$customReward = function($env,$stepCount,$state,$reward,$done,$info) use ($work) {
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

// { $numIterations = 1000; $logInterval = 1; $evalInterval = 10; $numEvalEpisodes = 10;
//   $maxExperienceSize = 50000; $batchSize = 32; $gamma = 0.99; $convLayers = null; $convType = null;
//   $fcLayers = [16,16,16]; $activation = null; $targetUpdatePeriod = 10; $targetUpdateTau = 1e-2;
//   $learningRate = 1e-3; $epsStart = 0.1; $epsStop = 0.1; $decayRate = 0.0005; $ddqn = false;
//   $lossFn = $nn->losses->MeanSquaredError(); $episodeAnnealing = true;
//   // no custom reward
// }

$numIterations = 1000;#300;#20000;#
$logInterval =   1;  #200;  #10; #
$evalInterval =  1;#10; #1000; #
$numEvalEpisodes = 0;#10;
$maxExperienceSize = 50000;#20000;#100000;
$batchSize = 32;#256;#128;#64;#
$gamma = 0.99;#1.0;#
$convLayers = null;
$convType = null;
$fcLayers = [16,16,16]; # [128,52];# [24,48];# [128,64,32];# [32,32];#
$activation = null;#'tanh';
$targetUpdatePeriod = 10;  #-1; #5;    #5;   #5;   #200;#
$targetUpdateTau =    1e-2;#1.0;#0.025;#0.01;#0.05;#1.0;#
$learningRate = 1e-3;#1e-5;#
$epsStart = 0.1;#1.0;#1.0; #1.0; #0.9;#1.0; #
$epsStop =  0.1;#0.1;#0.05;#0.01;#0.1;#0.05;#
$decayRate = 0.0005;#0.01;#0.001;#
$ddqn = true;#false;# 
$lossFn = $nn->losses->MeanSquaredError();
$episodeAnnealing = true;

$env = new MountainCarV0($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

[$state,$info] = $env->reset();
echo $mo->toString($state,'%1.2f')."\n";
//$env->render();
//$env->show();
//exit();

$evalEnv = new MountainCarV0($la);
$network = new QNetwork($la,$nn,$stateShape,$numActions,
    convLayers:$convLayers,convType:$convType,fcLayers:$fcLayers,activation:$activation);
$policy = new AnnealingEpsGreedy($la,$network,
    start:$epsStart,stop:$epsStop,decayRate:$decayRate,episodeAnnealing:$episodeAnnealing);
$dqnAgent = new DQN(
    $la,network:$network,policy:$policy,
    batchSize:$batchSize,gamma:$gamma,
    targetUpdatePeriod:$targetUpdatePeriod,targetUpdateTau:$targetUpdateTau,
    ddqn:$ddqn,lossFn:$lossFn,optimizerOpts:['lr'=>$learningRate],
    mo:$mo
);
$dqnAgent->summary();
$driver = new EpisodeRunner($la,$env,$dqnAgent,$maxExperienceSize);
//$driver = new StepRunner($la,$env,$dqnAgent,$maxExperienceSize,$evalEnv);
$driver->setCustomRewardFunction($customReward);
$filename = __DIR__.'\\mountaincar-dqn.model';
if(!file_exists($filename)) {
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:$maxSteps=null,
        metrics:$metrics=['steps','reward','epsilon','loss','val_steps','val_reward'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,verbose:$verbose=1);
    echo "\n";
    $ep = $mo->arange((int)($numIterations/$evalInterval),$evalInterval,$evalInterval);
    $arts[] = $plt->plot($ep,$la->scal(-1,$la->array($history['steps'])))[0];
    //$arts[] = $plt->plot($ep,$la->scal(-1,$la->array($history['val_steps'])))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    //$arts[] = $plt->plot($ep,$la->array($history['val_reward']))[0];
    //$arts[] = $plt->plot($ep,$la->increment($la->array($history['loss']),-100,100/max($history['loss'])))[0];
    //$arts[] = $plt->plot($ep,$la->increment($la->array($history['epsilon']),-100,100))[0];

    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    //$plt->legend($arts,['steps','val_steps','reward','val_reward','loss','epsilon']);
    //$plt->legend($arts,['reward','loss','val_reward','epsilon']);
    //$plt->legend($arts,['steps','val_steps']);
    $plt->legend($arts,['steps','reward']);
    $plt->show();
    $dqnAgent->saveWeightsToFile($filename);
} else {
    $dqnAgent->loadWeightsFromFile($filename);
}


echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    echo ".";
    [$state,$info] = $env->reset();
    $env->render();
    $done=false;
    while(!$done) {
        $action = $dqnAgent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $env->render();
    }
}
echo "\n";
$env->show();

print_r($work->getArrayCopy());