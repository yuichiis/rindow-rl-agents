<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Agent\QLearning\QLearning;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use function Rindow\Math\Matrix\R;

$mo = new MatrixOperator();
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$env = new CartPoleV1($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();

$numDizitized = 6;
$numStates = $numDizitized**$stateShape[0];

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
$customRewardFunc = function($env,$stepCount,$state,$action,$nextState,$reward,$done,$truncated,$info) use ($work) {
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
$decayRate=5e-5;
$eta=0.1;
$gamma=0.9;

//$poleRules = $la->ones($la->alloc([$numStates,$numActions]));

$policy = new AnnealingEpsGreedy($la,start:$espstart,stop:$espstop,decayRate:$decayRate);
$qlearning = new QLearning($la,$numStates,$numActions,$policy,$eta,$gamma,mo:$mo);
//$qlearning->setCustomStateFunction($digitizeState);
$driver3 = new EpisodeRunner($la,$env,$qlearning,$experienceSize=1);
//$driver3->setCustomRewardFunction($customRewardFunc);
$driver3->setCustomStateFunction($digitizeStateFunc);
#$agents = [$agent1,$agent2];
$drivers = [$driver3];

$arts = [];
foreach ($drivers as $driver) {
    $driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$episodes,metrics:['steps','val_steps','epsilon'],
        evalInterval:50,numEvalEpisodes:10,verbose:1);
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
    $testSteps = 0;
    while(!($done||$truncated)) {
        $action = $qlearning->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $state = $digitizeStateFunc($env,$state,$done);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
$filename = $env->show(path:__DIR__.'\\cartpole-ql-trained.gif');
echo "filename: {$filename}\n";
