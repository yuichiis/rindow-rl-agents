<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;
use Rindow\RL\Agents\Agent\DQN\Runner;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

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

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$seed = (int)(getenv('RL_SEED') ?: 1234);
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

// Evaluation must use exactly the same maze topology as training.
$env = new Maze(
    $la,policy:null,width:WIDTH,height:HEIGHT,exit:EXIT_STATE,
    throwInvalidAction:true,maxEpisodeSteps:MAX_EPISODE_STEPS,
);
$evalEnv = new Maze(
    $la,policy:$env->mazeRules(),width:WIDTH,height:HEIGHT,exit:EXIT_STATE,
    throwInvalidAction:true,maxEpisodeSteps:MAX_EPISODE_STEPS,
);
$env->actionSpace()->seed($seed);
$env->observationSpace()->seed($seed);
$evalEnv->actionSpace()->seed($seed+1);
$evalEnv->observationSpace()->seed($seed+1);

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
$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
    printf("Evaluation reward: %.1f\n",$runner->evaluate(EVAL_EPISODES));
} else {
    $history = $runner->train(
        $totalSteps,LEARNING_STARTS,TRAIN_EVERY,$evalEvery,EVAL_EPISODES,
        EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,$modelFile
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $la->array($history['step']),$la->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art],['DQN with action mask']);
        $plt->show(filename:__DIR__.'/../graphics/maze-dqn-history.png');
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
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
