<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV0;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Agents\Agent\QLearning;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\Network\QTable;

$mo = new MatrixOperator();
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$numDizitized = 6;
$cartPosBins  = $la->array(range(-2.4, 2.4, (2.4*2)/$numDizitized))[[1,$numDizitized-2]];
$cartVelocity = $la->array(range(-3.0, 3.0, (3.0*2)/$numDizitized))[[1,$numDizitized-2]];
$poleAngle    = $la->array(range(-0.5, 0.5, (0.5*2)/$numDizitized))[[1,$numDizitized-2]];
$poleVelocity = $la->array(range(-2.0, 2.0, (2.0*2)/$numDizitized))[[1,$numDizitized-2]];
$digitize = $la->stack([$cartPosBins,$cartVelocity,$poleAngle,$poleVelocity],$axis=0);
$digitizeState = function($env,$observation,$done) use ($la,$digitize,$numDizitized) {
    $state = 0;
    foreach($digitize as $idx => $thresholds) {
        $st = $la->searchsorted($thresholds,$observation[[$idx,$idx]]);
        $state *= $numDizitized;
        $state += $st->toArray()[0];
    }
    return $state;
};
$customReward = function($env,$stepCount,$observation,$reward,$done,$info) {
    if($done) {
        if($stepCount < 195) {
            $reward = -200.0;  // Episode failure
        } else {
            $reward = 1.0;     // Episode success
        }
    } else {
        $reward = 1.0; // reward each step
    }
    return $reward;
};
$poleRules = $la->ones($la->alloc([$numDizitized**4,2]));
[$width,$height,$exit] = [3,3,8];

$episodes = 2000;
$espstart=1.0;
$espstop=0.05;
$decayRate=0.0001;
$eta=0.1;
$gamma=0.9;

$env = new CartPoleV0($la);
$qtable = new QTable($la,$poleRules);
$policy = new AnnealingEpsGreedy($la,$qtable,$espstart,$espstop,$decayRate);
$qlearning = new QLearning($la,$qtable,$policy,$eta,$gamma,$mo);
//$qlearning->setDigitizeStateFunction($digitizeState);
$driver3 = new EpisodeDriver($la,$env,$qlearning,$experienceSize=1);
$driver3->setCustomRewardFunction($customReward);
$driver3->setCustomObservationFunction($digitizeState);
#$agents = [$agent1,$agent2];
$drivers = [$driver3];

$arts = [];
foreach ($drivers as $driver) {
    $driver->agent()->initialize();
    $history = $driver->train(
        $episodes,null,$metrics=['steps','val_steps','epsilon'],
        $evalInterval=50,$numEvalEpisodes=10,null,$verbose=1);
    $arts[] = $plt->plot($la->array($history['steps']))[0];
    $arts[] = $plt->plot($la->array($history['val_steps']))[0];
}
$plt->xlabel('Episode');
$plt->ylabel('Reward');
$plt->legend($arts,['steps','var_steps']);
$plt->show();

echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    echo ".";
    $observation = $env->reset();
    $observation = $digitizeState($env,$observation,false);
    $env->render();
    $done=false;
    while(!$done) {
        $action = $qlearning->action($observation,$training=false);
        [$observation,$reward,$done,$info] = $env->step($action);
        $observation = $digitizeState($env,$observation,$done);
        $env->render();
    }
}
echo "\n";
$env->show();
