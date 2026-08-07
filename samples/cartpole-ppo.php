<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;

const SEED = 42;
const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 1024;
const BATCH_SIZE = 256;
const EPOCHS = 4;
const GAMMA = 0.99;
const GAE_LAMBDA = 0.95;
const LEARNING_RATE = 1.0e-4;
const CLIP_RANGE = 0.2;
const VALUE_LOSS_WEIGHT = 0.5;
const ENTROPY_WEIGHT = 0.05;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = 475.0;
const MODEL_FILE = __DIR__.'/../models/cartpole-ppo.weights';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$env = new CartPoleV1($la);
$evalEnv = new CartPoleV1($la);
rlSeedSpaces($env,$evalEnv,$seed);

$agent = new PPOAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[64, 64],
    learningRate:LEARNING_RATE,
    clipRange:CLIP_RANGE,
    valueLossWeight:VALUE_LOSS_WEIGHT,
    entropyWeight:ENTROPY_WEIGHT,
    epochs:EPOCHS,
    batchSize:BATCH_SIZE,
    maxGradNorm:0.5,
);
$agent->summary();
$runner = new Runner(
    $la, $env, $evalEnv, $agent, ROLLOUT_STEPS, GAMMA, GAE_LAMBDA,
    SOLVED_REWARD
);
$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train($totalSteps, $evalEvery, $evalEpisodes);
    if (count($history['step']) > 0) {
        $steps = $la->array($history['step']);
        $art = $plt->plot($steps, $la->array($history['evalReward']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art], ['PPO']);
        $plt->show(filename:__DIR__.'/../graphics/cartpole-ppo-history.png');
    }
    $agent->saveWeightsToFile($modelFile);
    echo "Model saved: {$modelFile}\n";
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
    for ($episode = 1; $episode <= 5; $episode++) {
        [$obs] = $env->reset();
        $done = false;
        $totalReward = 0.0;
        $steps = 0;
        $env->render();
        while (!$done) {
            $action = $la->array($agent->selectActionDeterministic($obs), dtype:NDArray::int32);
            [$obs, $reward, $terminated, $truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $totalReward += $reward;
            $steps++;
            $env->render();
        }
        echo "Test Episode {$episode}, Steps: {$steps}, Total Reward: {$totalReward}\n";
    }
    $filename = $env->show(path:__DIR__.'/../graphics/cartpole-ppo-trained.gif');
    echo "filename: {$filename}\n";
}
