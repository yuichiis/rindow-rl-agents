<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;
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


$numIterations = 300000;#200000;#300;#1000;#
$targetScore = null; # -250; #
$numAchievements = null; # 10; #
$logInterval =   100;  #10; #
$evalInterval = 4000; #10; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;#100000;
$batchSize = 5; #  <= Pendulum 8 # 32;
//$gamma = 0.99;
$gamma = 0.9;#  # <= Pendulum 0.9; # 0.99;
$valueLossWeight = 0.5;
//$entropyWeight = 0.0;
$entropyWeight = 0.001; # <= Pendulum 0.001 # default:same 0.001 
//$entropyWeight = 0.01
$useBaseline = false; # false;
$useNormalize = true; # true;
$fcLayers = [128,128]; # [64,64];# [32,32];#
$initialStd = 4.5; # <= Pendulum 4.5; # 1.0
$learningRate = 7e-4;#1e-3;#1e-5;#
$minval = -0.003;
$maxval = 0.003;
$actionKernelInitializer = $nn->backend()->getInitializer('random_uniform',minval:$minval,maxval:$maxval);

echo "numIterations = {$numIterations}\n"; #
echo "targetScore = {$targetScore}\n"; #
echo "numAchievements = {$numAchievements}\n"; #
echo "logInterval = {$logInterval}\n"; #
echo "evalInterval = {$evalInterval}\n"; #
echo "numEvalEpisodes = {$numEvalEpisodes}\n"; #
echo "maxExperienceSize = {$maxExperienceSize}\n"; #
echo "batchSize = {$batchSize}\n"; #
echo "gamma = {$gamma}\n"; #
echo "valueLossWeight = {$valueLossWeight}\n"; #
echo "entropyWeight = {$entropyWeight}\n"; #
echo "useBaseline = {$useBaseline}\n"; #
echo "useNormalize = {$useNormalize}\n"; #
echo "fcLayers = [".implode(",",$fcLayers)."]\n"; #
echo "initialStd = {$initialStd}\n"; #
echo "learningRate = {$learningRate}\n"; #


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
$agent = new A2C(
    $la,
    continuous:true,
    nn:$nn,stateShape:$stateShape,actionSpace:$actionSpace,
    fcLayers:$fcLayers,
    actionKernelInitializer:$actionKernelInitializer,
    batchSize:$batchSize,gamma:$gamma,
    valueLossWeight:$valueLossWeight,entropyWeight:$entropyWeight,
    useBaseline:$useBaseline,useNormalize:$useNormalize,
    optimizerOpts:['lr'=>$learningRate],
    initialStd:$initialStd,
    mo:$mo,
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

$filename = __DIR__.'\\pendulum-a2c';
if(!$agent->fileExists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$agent,,experienceSize:$maxExperienceSize);
    $driver = new StepRunner($la,$env,$agent,experienceSize:$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $driver->metrics()->format('reward','%7.1f');
    $driver->metrics()->format('Ploss','%+5.2e');
    $driver->metrics()->format('Vloss','%+5.2e');
    $history = $driver->train(
        numIterations:$numIterations,maxSteps:null,
        //metrics:['steps','reward','Ploss','Vloss','entropy','std','valSteps','valRewards'],
        metrics:['reward','Ploss','Vloss','entropy','std','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,targetScore:$targetScore,numAchievements:$numAchievements,verbose:1,
    );
    $iter = $la->array($history['iter']);
    //$arts[] = $plt->plot($iter,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($iter,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['Ploss'],200,0))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['Vloss'],200,0))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['entropy'],200,0))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['std'],200,0))[0];
    //$arts[] = $plt->plot($iter,$la->array($history['valSteps']))[0];
    $arts[] = $plt->plot($iter,$la->array($history['valRewards']))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    #$plt->legend($arts,['steps','reward','epsilon','loss','valSteps','valReward']);
    //$plt->legend($arts,['reward','Ploss','Vloss','entropy','valRewards']);
    $plt->legend($arts,['reward','Ploss','Vloss','entropy','std','valRewards']);
    //$plt->legend($arts,['steps','valSteps']);
    $plt->show();
    //$agent->saveWeightsToFile($filename);
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
$filename = $env->show(path:__DIR__.'\\pendulum-a2c-trained.gif');
echo "filename: {$filename}\n";
