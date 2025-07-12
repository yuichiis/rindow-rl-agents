<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV0;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Agent\QLearning\QLearning;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use function Rindow\Math\Matrix\R;

$mo = new MatrixOperator();
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$env = new CartPoleV0($la);
$numActions = $env->actionSpace()->n();

$numDizitized = 6;
$cartPosBins  = $la->array(range(-2.4, 2.4, (2.4*2)/$numDizitized))[R(1,$numDizitized-1)];
$cartVelocity = $la->array(range(-3.0, 3.0, (3.0*2)/$numDizitized))[R(1,$numDizitized-1)];
$poleAngle    = $la->array(range(-0.5, 0.5, (0.5*2)/$numDizitized))[R(1,$numDizitized-1)];
$poleVelocity = $la->array(range(-2.0, 2.0, (2.0*2)/$numDizitized))[R(1,$numDizitized-1)];
$digitize = $la->stack([$cartPosBins,$cartVelocity,$poleAngle,$poleVelocity],$axis=0);
$digitizeStateFunc = function($env,$state,$done) use ($la,$digitize,$numDizitized) {
    $dizitizedState = 0;
    foreach($digitize as $idx => $thresholds) {
        $st = $la->searchsorted($thresholds,$state[R($idx,$idx+1)]);
        $dizitizedState *= $numDizitized;
        $dizitizedState += $st->toArray()[0];
    }
    return $la->array([$dizitizedState],dtype:NDArray::int32);
};
$customRewardFunc = function($env,$stepCount,$state,$reward,$done,$info) {
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

$episodes = 2000;
$espstart=1.0;
$espstop=0.05;
$decayRate=0.0001;
$eta=0.1;
$gamma=0.9;

$poleRules = $la->ones($la->alloc([$numDizitized**4,$numActions]));

$policy = new AnnealingEpsGreedy($la,start:$espstart,stop:$espstop,decayRate:$decayRate);
$qlearning = new QLearning($la,$poleRules,$policy,$eta,$gamma,mo:$mo);
//$qlearning->setCustomStateFunction($digitizeState);
$driver3 = new EpisodeRunner($la,$env,$qlearning,$experienceSize=1);
$driver3->setCustomRewardFunction($customRewardFunc);
$driver3->setCustomStateFunction($digitizeStateFunc);
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
    [$state,$info] = $env->reset();
    $state = $digitizeStateFunc($env,$state,false);
    $env->render();
    $done=false;
    $truncated=false;
    $testReward = 0;
    while(!($done||$truncated)) {
        $action = $qlearning->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $state = $digitizeStateFunc($env,$state,$done);
        $testReward += $reward;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Total Reward: {$testReward}\n";
}
echo "\n";
$env->show();
