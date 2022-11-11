<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Agents\Agent\Reinforce;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\RL\Agents\Network\QNetwork;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$mazeRules = $la->array([
//   UP    DOWN  RIGHT LEFT
    [NAN,    1,    1,  NAN], // 0  +-+-+-+
    [NAN,    1,    1,    1], // 1  |0 1 2|
    [NAN,  NAN,  NAN,    1], // 2  + + +-+
    [  1,    1,  NAN,  NAN], // 3  |3|4 5|
    [  1,  NAN,    1,  NAN], // 4  + +-+ +
    [NAN,    1,  NAN,    1], // 5  |6 7|8|
    [  1,  NAN,    1,  NAN], // 6  +-+-+-+
    [NAN,  NAN,  NAN,    1], // 7
    [  1,  NAN,  NAN,  NAN], // 8
]);
$customObservationFunction = function($env,$observation,$done) use ($la) {
    return $la->array([$observation],NDArray::float32);
};

[$width,$height,$exit] = [3,3,8];
$maxEpisodeSteps = 100;
$episodes = 300;
$evalInterval = 10;
$numEvalEpisodes = 1;
$maxExperienceSize = 10000;#100000;
#######################
$batchSize = 32;
$gamma = 0.9;#0.99;#1.0;#
$fcLayers = [10,10];# [32,32];#
$targetUpdatePeriod = 5;   #5;  #5;    #5;   #200;#
$targetUpdateTau =    0.05;#0.1;#0.025;#0.01;#1.0;#
$learningRate = 1e-3;#1e-1;#
$epsStart = 0.9;#0.1;#1.0; #1.0; #1.0; #
$epsStop =  0.1;#0.1;#0.05;#0.01;#0.05;#
$decayRate = 0.07;#0.001;#0.0005;#
$episodeAnnealing = true;
$experienceSize = 10000;#100;#
$activation = 'tanh';#'relu';#
$lossFn = $nn->losses->MeanSquaredError();
$useBaseline = false;

$env = new Maze($la,$mazeRules,$width,$height,$exit,$throw=true,$maxEpisodeSteps);
//$env->reset();
//$env->render();
//$env->show();
//exit();

//$obsSize = $env->observationSpace()->shape();
$obsSize = [1];
$numActions = $env->actionSpace()->n();

$network = new QNetwork($la,$nn,$obsSize,$numActions,$convLayers=null,$convType=null,$fcLayers,$activation,null,$mazeRules);
//$network = new QNetwork($la,$nn,$obsSize,$numActions,$fcLayers,$activation,null,$mazeRules,null,$mo);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$decayRate);
$policy = new Boltzmann($la,$network,softmax:true);
$agent = new Reinforce(
    $la,network:$network,policy:$policy,gamma:$gamma,useBaseline:$useBaseline,
    optimizerOpts:['lr'=>$learningRate],mo:$mo,
);
$agent->summary();

$driver4 = new EpisodeDriver($la,$env,$agent,$experienceSize,null,$episodeAnnealing);
$driver4->setCustomObservationFunction($customObservationFunction);
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
        $arts[] = $plt->plot($ep,$la->scal(-1,$la->array($history['reward'])))[0];
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
        $observation = $env->reset();
        $observation = $customObservationFunction($env,$observation,false);
        $env->render();
        $done=false;
        while(!$done) {
            $action = $agent->action($observation,$training=false);
            [$observation,$reward,$done,$info] = $env->step($action);
            $observation = $customObservationFunction($env,$observation,$done);
            $env->render();
        }
    }
}
echo "\n";
$env->show(null,$delay=100);
