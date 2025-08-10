<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;
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


$numIterations = 300000;# 100000;#300;#1000;#
$targetScore = null; # -250; #
$numAchievements = null; # 10; #
$logInterval =   null; #1000;  #10; #
$evalInterval =  1024; #10; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;#100000;
$rolloutSteps = 1024; # 2048;
$batchSize = 64;# 32;#
$epochs = 10;
$gamma = 0.9;# 0.9;# 0.99;
$gaeLambda = 0.95;
$valueLossWeight = 0.5;
$entropyWeight = 0.01;
$fcLayers = [128,128];
$normAdv = true;
$clipEpsilon = 0.2;
$clipValueLoss = true; # false; # 
$learningRate = 3e-4;# 1e-3;#1e-5;#
$clipnorm = 0.5;
// Initializing the action layer kernel for Pendulum
$minval = -0.003;
$maxval = 0.003;
$actionKernelInitializer = $nn->backend()->getInitializer('random_uniform',minval:$minval,maxval:$maxval);


$env = new PendulumV1($la);
$stateShape = $env->observationSpace()->shape();
$actionSpace = $env->actionSpace();

//$env->reset();
//$env->render();
//$env->show();
//exit();

$evalEnv = new PendulumV1($la);
//$network = new QNetwork($la,$nn,$stateShape,$numActions,$convLayers,$convType,$fcLayers);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$epsDecayRate);
$agent = new PPO(
    $la,
    continuous:true,
    nn:$nn,stateShape:$stateShape,actionSpace:$actionSpace,
    rolloutSteps:$rolloutSteps,epochs:$epochs,batchSize:$batchSize,
    fcLayers:$fcLayers,
    actionKernelInitializer:$actionKernelInitializer,
    gamma:$gamma,gaeLambda:$gaeLambda,
    valueLossWeight:$valueLossWeight,entropyWeight:$entropyWeight,
    normAdv:$normAdv,
    clipEpsilon:$clipEpsilon,clipValueLoss:$clipValueLoss,
    clipnorm:$clipnorm,
    optimizerOpts:['lr'=>$learningRate],mo:$mo,
);
$agent->summary();

$filename = __DIR__.'\\pendulum-ppo';
if(!$agent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$agent,,experienceSize:$maxExperienceSize);
    $driver = new StepRunner($la,$env,$agent,experienceSize:$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $driver->metrics()->format('reward','%7.1f');
    $driver->metrics()->format('Ploss','%+5.2e');
    $driver->metrics()->format('Vloss','%+5.2e');
    $driver->metrics()->format('actmin','%+6.3f');
    $driver->metrics()->format('actmax','%+6.3f');
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:null,
        //metrics:['steps','reward','Ploss','Vloss','entropy','std','valSteps','valRewards'],
        metrics:['reward','Ploss','Vloss','std','actmin','actmax','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,targetScore:$targetScore,numAchievements:$numAchievements,verbose:1,
    );
    $ep = $la->array($history['iter']);
    //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,$la->scal(200/max($history['Ploss']),$la->array($history['Ploss'])))[0];
    $arts[] = $plt->plot($ep,$la->scal(200/max($history['Vloss']),$la->array($history['Vloss'])))[0];
    //$arts[] = $plt->plot($ep,$la->scal(200/max($history['entropy']),$la->array($history['entropy'])))[0];
    $arts[] = $plt->plot($ep,$la->scal(200/max($history['std']),$la->array($history['std'])))[0];
    //$arts[] = $plt->plot($ep,$la->array($history['valSteps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valRewards']))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    #$plt->legend($arts,['steps','reward','epsilon','loss','valSteps','valReward']);
    //$plt->legend($arts,['reward','Ploss','Vloss','entropy','valRewards']);
    $plt->legend($arts,['reward','Ploss','Vloss','std','valRewards']);
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
$filename = $env->show(path:__DIR__.'\\pendulum-ppo-trained.gif');
echo "filename: {$filename}\n";
