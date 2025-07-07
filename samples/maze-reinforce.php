<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Agent\Reinforce\Reinforce;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

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
$customStateFunction = function($env,$x,$done) use ($la) {
    return $la->expandDims($x,axis:-1);
};

[$width,$height,$exit] = [3,3,8];
$maxEpisodeSteps = 100;
$episodes = 300;
$evalInterval = 10;
$numEvalEpisodes = 1;
$maxExperienceSize = 10000;#100000;
#######################
//$batchSize = 32;
$gamma = 0.99;#0.9;#1.0;#
$fcLayers = [100];# [128,128];# [10,10];#
$activation = 'relu';#'tanh';#
#$targetUpdatePeriod = 5;   #5;  #5;    #5;   #200;#
#$targetUpdateTau =    0.05;#0.1;#0.025;#0.01;#1.0;#
$learningRate = 1e-3;#1e-1;#
#$epsStart = 0.9;#0.1;#1.0; #1.0; #1.0; #
#$epsStop =  0.1;#0.1;#0.05;#0.01;#0.05;#
#$decayRate = 0.07;#0.001;#0.0005;#
#$episodeAnnealing = true;
$experienceSize = 10000;#100;#
//$lossFn = $nn->losses->MeanSquaredError();
$useBaseline = true;

$env = new Maze($la,$mazeRules,$width,$height,$exit,$throw=true,$maxEpisodeSteps);
//$env->reset();
//$env->render();
//$env->show();
//exit();

//$stateShape = $env->observationSpace()->shape();
$stateShape = [1];
$numActions = $env->actionSpace()->n();

//$network = new QNetwork($la,$nn,$stateShape,$numActions,$convLayers=null,$convType=null,$fcLayers,$activation,null,$mazeRules);
//$network = new QNetwork($la,$nn,$stateShape,$numActions,$fcLayers,$activation,null,$mazeRules,null,$mo);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$decayRate);
//$policy = new Boltzmann($la);
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

$driver4 = new EpisodeRunner($la,$env,$agent,$experienceSize);
$driver4->setCustomStateFunction($customStateFunction);
$drivers = [$driver4];

$filenamePattern = __DIR__.'\\maze-reinforce-%d';
$arts = [];
foreach ($drivers as $i => $driver) {
    $agent = $driver->agent();
    $filename = sprintf($filenamePattern,$i);
    if(!$agent->fileExists($filename)) {
        // $agent->initialize();
        $history = $driver->train(
            $episodes,null,$metrics=['steps','reward','val_steps','epsilon'],
            $evalInterval,$numEvalEpisodes,null,$verbose=2);
        $ep = $mo->arange((int)floor($episodes/$evalInterval),$evalInterval,$evalInterval);
        $arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
        $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
        $arts[] = $plt->plot($ep,$la->array($history['val_steps']))[0];

        $plt->legend($arts,['steps','reward','val_steps']);
        $plt->xlabel('episodes');
        $plt->ylabel('avg steps');
        $plt->show();
        $agent->saveWeightsToFile($filename);
    } else {
        $agent->loadWeightsFromFile($filename);
    }
}

echo "Creating demo animation.\n";
foreach($drivers as $i => $driver) {
    $agent = $driver->agent();
    for($i=0;$i<1;$i++) {
        echo ".";
        [$state,$info] = $env->reset();
        $state = $customStateFunction($env,$state,false);
        $env->render();
        $done=false;
        $truncated=false;
        while(!($done || $truncated)) {
            $action = $agent->action($state,training:false,info:$info);
            [$state,$reward,$done,$truncated,$info] = $env->step($action);
            $state = $customStateFunction($env,$state,$done);
            $env->render();
        }
    }
}
echo "\n";
$env->show(null,$delay=100);
