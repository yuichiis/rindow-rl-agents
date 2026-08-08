<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\Sarsa\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Agents\Agent\Sarsa\TrueOnlineSarsaLambdaAgent;
use Rindow\RL\Agents\Env\MountainCar\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const SEED = 42;
const TOTAL_EPISODES = 1000;
const NUM_TILINGS = 8;
const TILES_PER_DIMENSION = 8;
const LEARNING_RATE = 0.3;
const GAMMA = 1.0;
const LAMBDA = 0.9;
const EPSILON = 0.0;
const EVAL_EVERY = 25;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = -110.0;
const MODEL_FILE = __DIR__.'/../models/mountaincar-true-online-sarsa-lambda.weights';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$env = new MountainCarV0($hostLa);
$evalEnv = new MountainCarV0($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
}

$tileCoder = new TileCoder(
    low:[-1.2, -0.07],
    high:[0.6, 0.07],
    numTilings:NUM_TILINGS,
    tilesPerDimension:TILES_PER_DIMENSION,
);
$agent = new TrueOnlineSarsaLambdaAgent(
    $la,
    $tileCoder,
    numActions:$env->actionSpace()->n(),
    learningRate:LEARNING_RATE,
    gamma:GAMMA,
    lambda:LAMBDA,
    epsilon:EPSILON,
    nn:$nn,
);
$runner = new Runner($la, $env, $evalEnv, $agent, solvedReward:SOLVED_REWARD);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalEpisodes = rlEnvInt('RL_TOTAL_EPISODES',TOTAL_EPISODES);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    // True Online Sarsa(lambda) learns directly from MountainCar's raw -1 reward.
    $history = $runner->train(
        $totalEpisodes,
        $evalEvery,
        $evalEpisodes,
        bestModelFile:$modelFile,
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    } else {
        $agent->saveWeightsToFile($modelFile);
        echo "Model saved: {$modelFile}\n";
    }

    if (count($history['episode']) > 0) {
        $episodes = $hostLa->array($history['episode']);
        $trainArt = $plt->plot($episodes, $hostLa->array($history['trainReward']))[0];
        $evalArt = $plt->plot($episodes, $hostLa->array($history['evalReward']))[0];
        $plt->xlabel('Training episodes');
        $plt->ylabel('Raw reward');
        $plt->legend([$trainArt, $evalArt], ['Training raw reward', 'Evaluation raw reward']);
        $plt->show(filename:__DIR__.'/../graphics/mountaincar-true-online-sarsa-lambda-history.png');
    }
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
    for ($episode = 1; $episode <= 5; $episode++) {
        [$observation] = $env->reset();
        $done = false;
        $totalReward = 0.0;
        $steps = 0;
        $goal = false;
        $env->render();
        while (!$done) {
            $action = $la->array(
                $agent->selectActionDeterministic($observation), dtype:NDArray::int32
            );
            [$observation, $reward, $terminated, $truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $totalReward += $reward;
            $steps++;
            $goal = $goal || $terminated;
            $env->render();
        }
        printf("Test Episode %d | Steps=%d | RawReward=%+.1f | Goal=%s\n",
            $episode, $steps, $totalReward, $goal ? 'yes' : 'no');
    }
    $filename = $env->show(
        path:__DIR__.'/../graphics/mountaincar-true-online-sarsa-lambda-trained.gif'
    );
    echo "filename: {$filename}\n";
}
