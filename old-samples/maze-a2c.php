<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\A2C\A2C;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$maxEpisodeSteps = 100;
[$width,$height,$exit] = [3,3,8];
$mazeRules = $la->array([
//   UP    DOWN  RIGHT LEFT
    [false,  true,  true, false], // 0  +-+-+-+
    [false,  true,  true,  true], // 1  |0 1 2|
    [false, false, false,  true], // 2  + + +-+
    [ true,  true, false, false], // 3  |3|4 5|
    [ true, false,  true, false], // 4  + +-+ +
    [false,  true, false,  true], // 5  |6 7|8|
    [ true, false,  true, false], // 6  +-+-+-+
    [false, false, false,  true], // 7
    [ true, false, false, false], // 8
],dtype:NDArray::bool);

$numIterations = 50000;#200000;#300;#1000;#
$logInterval =   100;  #10; #
$evalInterval = 1000; #10; #
$numEvalEpisodes = 5;
$maxExperienceSize = 10000;#100000;
#######################
$batchSize = 32; # <= Maze 32 # 5; #
$gamma = 0.99; # 0.9;#1.0;#
$valueLossWeight = 0.5;
$entropyWeight = 0.001; # 0.01;
$fcLayers = [64];# [32,32];# [128,128];# [10,10];#
#$activation = 'relu';#'tanh';#
$learningRate = 7e-4; # 1e-1;#
$experienceSize = 10000;#100;#
//$lossFn = $nn->losses->MeanSquaredError();
$useBaseline = true; # <= Maze true # false #
$useNormalize = false; # <= Maze false # true #



$env = new Maze($la,$mazeRules,$width,$height,$exit,$throw=true,$maxEpisodeSteps);
//$env->reset();
//$env->render();
//$env->show();
//exit();

$stateShape = $env->observationSpace()['location']->shape();
//$numStates = $env->observationSpace()->n();
$numActions = $env->actionSpace()->n();
//$stateShape = [$numStates];
//$customStateFunction = function($env,$x,$done) use ($la,$numStates) {
//    $x = $la->expandDims($x,0);
//    $state = $la->onehot($x,$numStates);
//    $state = $la->squeeze($state,0);
//    return $state;
//};

$evalEnv = new Maze($la,$mazeRules,$width,$height,$exit,$throw=true,$maxEpisodeSteps);

//$network = new QNetwork($la,$nn,$stateShape,$numActions,$convLayers=null,$convType=null,$fcLayers,$activation,null,$mazeRules);
//$network = new QNetwork($la,$nn,$stateShape,$numActions,$fcLayers,$activation,null,$mazeRules,null,$mo);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$decayRate);
//$policy = new Boltzmann($la);
$agent = new A2C(
    $la,
    nn:$nn,stateShape:$stateShape,numActions:$numActions,fcLayers:$fcLayers,
    batchSize:$batchSize,gamma:$gamma,
    valueLossWeight:$valueLossWeight,entropyWeight:$entropyWeight,
    useBaseline:$useBaseline,useNormalize:$useNormalize,
    optimizerOpts:['lr'=>$learningRate],
    stateField:'location',mo:$mo,
);
//$agent->setCustomStateFunction($customStateFunction);
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

$filename = __DIR__.'\\maze-reinforce';
if(!$agent->fileExists($filename)) {
    $driver = new StepRunner($la,$env,$agent,experienceSize:$experienceSize,evalEnv:$evalEnv);
    $driver->metrics()->format('reward','%5.1f');
    $driver->metrics()->format('Ploss','%+5.2e');
    $driver->metrics()->format('Vloss','%+5.2e');
    $arts = [];
    // $agent->initialize();
    $history = $driver->train(
        numIterations:$numIterations,logInterval:$logInterval,
        metrics:['reward','Ploss','Vloss','entropy','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,verbose:1,
    );
    echo "\n";
    $ep = $la->array($history['iter']);
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['Ploss'],50,0))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['Vloss'],50,0))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['entropy'],50,0))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valRewards']))[0];
    $plt->legend($arts,['reward','Ploss','Vloss','entropy','valRewards']);
    $plt->xlabel('episodes');
    $plt->ylabel('avg steps');
    $plt->show();
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}

echo "Creating demo animation.\n";
for($i=0;$i<1;$i++) {
    echo ".";
    [$state,$info] = $env->reset();
    //$state = $customStateFunction($env,$state,false);
    $env->render();
    $done=false;
    $truncated=false;
    $testReward = 0;
    $testSteps = 0;
    while(!($done || $truncated)) {
        $action = $agent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        //$state = $customStateFunction($env,$state,$done);
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'\\maze-reinforce-trained.gif',delay:100);
echo "filename: {$filename}\n";
