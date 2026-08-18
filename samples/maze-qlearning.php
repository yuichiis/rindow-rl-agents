<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\QLearning\QLearningAgent;
use Rindow\RL\Agents\Env\Maze\DeviceWrapper;
use Rindow\RL\Agents\Agent\QLearning\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

const SEED = 1234;
const WIDTH=3, HEIGHT=3, EXIT_STATE=8, MAX_EPISODE_STEPS=100;
const TOTAL_EPISODES=500, EVAL_EVERY=10, EVAL_EPISODES=10;
const MODEL_FILE=__DIR__.'/../models/maze-qlearning.weights';
$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";
$mazeRules = $hostLa->array([
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

$env=new Maze(
    $hostLa,policy:$mazeRules,width:WIDTH,height:HEIGHT,exit:EXIT_STATE,
    throwInvalidAction:true,maxEpisodeSteps:MAX_EPISODE_STEPS
);
$evalEnv=new Maze(
    $hostLa,policy:$mazeRules,width:WIDTH,height:HEIGHT,exit:EXIT_STATE,
    throwInvalidAction:true,maxEpisodeSteps:MAX_EPISODE_STEPS
);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
}
$coder=new TileCoder([0.0,0.0],[HEIGHT-1.0,WIDTH-1.0],4,2);
$agent=new QLearningAgent($la,$coder,$env->actionSpace()->n(),0.2,1.0,0.1,
    stateField:'location',actionMaskField:'actionMask',nn:$nn);
$runner=new Runner($la,$env,$evalEnv,$agent);
$modelFile=rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
if(is_file($modelFile)) { $agent->loadWeightsFromFile($modelFile); echo "Model loaded: {$modelFile}\n"; }
else { $runner->train(rlEnvInt('RL_TOTAL_EPISODES',TOTAL_EPISODES),
    rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY),$evalEpisodes,$modelFile);
    if(is_file($modelFile)) $agent->loadWeightsFromFile($modelFile); }
if(!rlEnvBool('RL_SKIP_DEMO')) { [$obs]=$env->reset(); $done=false; $total=0; $steps=0; $env->render();
    while(!$done) { $action=$la->array($agent->selectActionDeterministic($obs),dtype:NDArray::int32);
        [$obs,$reward,$terminated,$truncated]=$env->step($action); $done=$terminated||$truncated;
        $total+=$reward; $steps++; $env->render(); }
    printf("Test Episode 1 | Steps=%d | RawReward=%+.1f\n",$steps,$total);
    echo 'filename: '.$env->show(path:__DIR__.'/../graphics/maze-qlearning-trained.gif',delay:100)."\n"; }
