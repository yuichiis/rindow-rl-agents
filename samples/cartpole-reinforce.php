<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\Reinforce\ReinforceAgent;
use Rindow\RL\Agents\Agent\Reinforce\Runner;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;

const SEED = 42;
const TOTAL_EPISODES = 2_000;
const GAMMA = 0.99;
const LEARNING_RATE = 1.0e-2;
const ENTROPY_WEIGHT = 0.0;
const EVAL_EVERY = 50;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = 475.0;
const MODEL_FILE = __DIR__.'/../models/cartpole-reinforce.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$seedText = getenv('RL_SEED');
$seed = $seedText === false ? SEED : (int)$seedText;
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
$la->setSeed($seed);
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);
$env = new CartPoleV1($la);
$evalEnv = new CartPoleV1($la);
$env->observationSpace()->seed($seed);
$env->actionSpace()->seed($seed);

$agent = new ReinforceAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128],
    learningRate:LEARNING_RATE,
    entropyWeight:ENTROPY_WEIGHT,
    maxGradNorm:1.0,
);
$agent->summary();
$runner = new Runner($la, $env, $evalEnv, $agent, GAMMA, true, SOLVED_REWARD);
$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalEpisodes = (int)(getenv('RL_TOTAL_EPISODES') ?: TOTAL_EPISODES);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train($totalEpisodes, $evalEvery, EVAL_EPISODES);
    if (count($history['episode']) > 0) {
        $art = $plt->plot(
            $la->array($history['episode']), $la->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training episodes');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art], ['REINFORCE']);
        $plt->show(filename:__DIR__.'/../graphics/cartpole-reinforce-history.png');
    }
    $agent->saveWeightsToFile($modelFile);
    echo "Model saved: {$modelFile}\n";
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    echo "Creating demo animation.\n";
    for ($episode = 1; $episode <= 5; $episode++) {
        [$observation] = $env->reset();
        $done = false;
        $totalReward = 0.0;
        $steps = 0;
        $env->render();
        while (!$done) {
            $action = $la->array(
                $agent->selectActionDeterministic($observation), dtype:NDArray::int32
            );
            [$observation, $reward, $terminated, $truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $totalReward += $reward;
            $steps++;
            $env->render();
        }
        echo "Test Episode {$episode}, Steps: {$steps}, Total Reward: {$totalReward}\n";
    }
    $filename = $env->show(path:__DIR__.'/../graphics/cartpole-reinforce-trained.gif');
    echo "filename: {$filename}\n";
}
