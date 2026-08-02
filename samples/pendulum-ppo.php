<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;

const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 1024;
const BATCH_SIZE = 64;
const EPOCHS = 10;
const GAMMA = 0.90;
const GAE_LAMBDA = 0.95;
const LEARNING_RATE = 3.0e-4;
const ENTROPY_WEIGHT = 0.01;
const EVAL_EVERY = 1024;
const EVAL_EPISODES = 10;
const MODEL_FILE = __DIR__.'/../models/pendulum-ppo.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$seedText = getenv('RL_SEED');
if ($seedText !== false) {
    $seed = (int)$seedText;
    $la->setSeed($seed);
    echo "Random seed: {$seed}\n";
} else {
    echo "Random seed: system default (set RL_SEED for reproducible runs)\n";
}

$env = new PendulumV1($la);
$evalEnv = new PendulumV1($la);
if ($seedText !== false) {
    $env->actionSpace()->seed($seed);
    $env->observationSpace()->seed($seed);
    $evalEnv->actionSpace()->seed($seed + 1);
    $evalEnv->observationSpace()->seed($seed + 1);
}

$actionSpace = $env->actionSpace();
$actionDim = $actionSpace->shape()[0];
$agent = new PPOAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$actionDim,
    hiddenLayers:[128, 128],
    learningRate:LEARNING_RATE,
    clipRange:0.2,
    valueLossWeight:0.5,
    entropyWeight:ENTROPY_WEIGHT,
    epochs:EPOCHS,
    batchSize:BATCH_SIZE,
    maxGradNorm:0.5,
    clipValueLoss:true,
    sharedBackbone:true,
    continuous:true,
    actionMin:$actionSpace->low(),
    actionMax:$actionSpace->high(),
);
$agent->summary();

$runner = new Runner(
    $la,
    $env,
    $evalEnv,
    $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:-200.0,
    bootstrapTruncated:false,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
} else {
    $runner->train($totalSteps, $evalEvery, EVAL_EPISODES, bestModelFile:$modelFile);
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    echo "Creating demo animation.\n";
    for ($episode = 1; $episode <= 5; $episode++) {
        [$obs] = $env->reset();
        $done = false;
        $total = 0.0;
        $steps = 0;
        $env->render();
        while (!$done) {
            $action = $agent->selectActionDeterministic($obs);
            [$obs, $reward, $terminated, $truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $total += $reward;
            $steps++;
            $env->render();
        }
        printf("Test Episode %d, Steps: %d, Total Reward: %.1f\n", $episode, $steps, $total);
    }
    $filename = $env->show(path:__DIR__.'/../graphics/pendulum-ppo-trained.gif');
    echo "filename: {$filename}\n";
}
