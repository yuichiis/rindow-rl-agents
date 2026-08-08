<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\Reinforce\ReinforceAgent;
use Rindow\RL\Agents\Agent\Reinforce\Runner;
use Rindow\RL\Agents\Env\CartPole\DeviceWrapper;
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

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$env = new CartPoleV1($hostLa);
$evalEnv = new CartPoleV1($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn, $env);
    $evalEnv = new DeviceWrapper($nn, $evalEnv);
}

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
$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalEpisodes = rlEnvInt('RL_TOTAL_EPISODES',TOTAL_EPISODES);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train($totalEpisodes, $evalEvery, $evalEpisodes);
    if (count($history['episode']) > 0) {
        $art = $plt->plot(
            $hostLa->array($history['episode']), $hostLa->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training episodes');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art], ['REINFORCE']);
        $plt->show(filename:__DIR__.'/../graphics/cartpole-reinforce-history.png');
    }
    $agent->saveWeightsToFile($modelFile);
    echo "Model saved: {$modelFile}\n";
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
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
