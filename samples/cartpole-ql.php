<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Agent\QLearning\QLearning;
use Rindow\RL\Agents\Agent\Sarsa\Sarsa;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use function Rindow\Math\Matrix\R;

$mo = new MatrixOperator();
$la = $mo->la();
$plt = new Plot(null,$mo);

$env = new CartPoleV1($la);
$stateShape = $env->observationSpace()->shape();
$numActions = $env->actionSpace()->n();
$evalEnv = new CartPoleV1($la);

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

// == StepRunner ==
$numIterations = 100000;
$evalInterval = 1000;
$logInterval = null;
$numEvalEpisodes = 10;
//// == EpisodeRunner ==
//$numIterations = 1500;
//$evalInterval = 10;
//$logInterval = 1;
//$numEvalEpisodes = 5;

$espstart=1.0;
$espstop=0.05;
$decayRate=5e-5;
$eta=0.1;
$gamma=0.9;

$policy = new AnnealingEpsGreedy($la,start:$espstart,stop:$espstop,decayRate:$decayRate);
//$agent = new QLearning($la,$numStates,$numActions,$policy,$eta,$gamma,mo:$mo);
$agent = new Sarsa($la,$numStates,$numActions,$policy,$eta,$gamma,mo:$mo);
$agent->setCustomStateFunction($digitizeStateFunc);

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

$filename = __DIR__.'\\cartpole-ql';
if(!$agent->fileExists($filename)) {
    $driver = new StepRunner($la,$env,$agent,experienceSize:10,evalEnv:$evalEnv);
    //$driver = new EpisodeRunner($la,$env,$agent,experienceSize:10,evalEnv:$evalEnv);
    $driver->metrics()->format('reward','%5.1f');
    $driver->metrics()->format('valRewards','%5.1f');
    $arts = [];
    $driver->agent()->initialize();
    $history = $driver->train(
        numIterations:$numIterations,metrics:['reward','error','valRewards','epsilon'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,
        logInterval:$logInterval,verbose:1,
    );
    $iter = $la->array($history['iter']);
    $arts[] = $plt->plot($iter,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($iter,$la->array($history['valRewards']))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['error'],200,0))[0];
    $arts[] = $plt->plot($iter,fitplot($la,$history['epsilon'],200,0))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    $plt->legend($arts,['reward','valRewards','error','epsilon']);
    $plt->show();
    //$agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}

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
        $action = $agent->action($state,training:false,info:$info);
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
