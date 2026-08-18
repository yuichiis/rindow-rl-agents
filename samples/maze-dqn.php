<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;
use Rindow\RL\Agents\Agent\DQN\Runner;
use Rindow\RL\Agents\Env\Maze\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

const SEED = 1234;
const WIDTH = 3;
const HEIGHT = 3;
const EXIT_STATE = 8;
const MAX_EPISODE_STEPS = 100;
const TOTAL_STEPS = 10_000;
const BUFFER_SIZE = 10_000;
const BATCH_SIZE = 64;
const LEARNING_STARTS = 200;
const TRAIN_EVERY = 4;
const TARGET_UPDATE_INTERVAL = 250;
const GAMMA = 0.99;
const LEARNING_RATE = 1.0e-3;
const EPSILON_START = 1.0;
const EPSILON_END = 0.05;
const EPSILON_DECAY_STEPS = 5_000;
const EVAL_EVERY = 1_000;
const EVAL_EPISODES = 10;
const MODEL_FILE = __DIR__.'/../models/maze-dqn.weights';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);
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

$env = new Maze(
    $hostLa,policy:$mazeRules,width:WIDTH,height:HEIGHT,exit:EXIT_STATE,
    throwInvalidAction:true,maxEpisodeSteps:MAX_EPISODE_STEPS,
);
$evalEnv = new Maze(
    $hostLa,policy:$mazeRules,width:WIDTH,height:HEIGHT,exit:EXIT_STATE,
    throwInvalidAction:true,maxEpisodeSteps:MAX_EPISODE_STEPS,
);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
}

$agent = new DQNAgent(
    $nn,
    obsDim:$env->observationSpace()['location']->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128,128],
    learningRate:LEARNING_RATE,
    gamma:GAMMA,
    batchSize:BATCH_SIZE,
    targetUpdateInterval:TARGET_UPDATE_INTERVAL,
    maxGradNorm:10.0,
    stateField:'location',
    actionMaskField:'actionMask',
);
$agent->summary();

$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    obsDim:$env->observationSpace()['location']->shape()[0],
    bufferSize:BUFFER_SIZE,
);
$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);

if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
    printf("Evaluation reward: %.1f\n",$runner->evaluate($evalEpisodes));
} else {
    $history = $runner->train(
        $totalSteps,LEARNING_STARTS,TRAIN_EVERY,$evalEvery,$evalEpisodes,
        EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,$modelFile
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $hostLa->array($history['step']),$hostLa->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art],['DQN with action mask']);
        $plt->show(filename:__DIR__.'/../graphics/maze-dqn-history.png');
    }
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
    [$observation] = $env->reset();
    $env->render();
    $done = false;
    $totalReward = 0.0;
    $steps = 0;
    while (!$done) {
        $action = $la->array(
            $agent->selectActionDeterministic($observation),dtype:NDArray::int32
        );
        [$observation,$reward,$terminated,$truncated] = $env->step($action);
        $done = $terminated || $truncated;
        $totalReward += $reward;
        $steps++;
        $env->render();
    }
    printf("Test Episode 1, Steps: %d, Total Reward: %.1f\n",$steps,$totalReward);
    $filename = $env->show(path:__DIR__.'/../graphics/maze-dqn-trained.gif',delay:100);
    echo "filename: {$filename}\n";
}
