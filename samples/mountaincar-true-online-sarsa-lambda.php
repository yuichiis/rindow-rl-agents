<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Agent\TileCoding\Runner;
use Rindow\RL\Agents\Agent\TileCoding\TileCoder;
use Rindow\RL\Agents\Agent\TileCoding\TrueOnlineSarsaLambdaAgent;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

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

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$seedText = getenv('RL_SEED');
$seed = $seedText === false ? null : (int)$seedText;
if ($seed !== null) {
    $la->setSeed($seed);
    echo "Random seed: {$seed}\n";
} else {
    echo "Random seed: system default (set RL_SEED for reproducible runs)\n";
}

$env = new MountainCarV0($la);
$evalEnv = new MountainCarV0($la);
if ($seed !== null) {
    $env->observationSpace()->seed($seed);
    $env->actionSpace()->seed($seed);
    $evalEnv->observationSpace()->seed($seed + 1);
    $evalEnv->actionSpace()->seed($seed + 1);
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
);
$runner = new Runner($la, $env, $evalEnv, $agent, solvedReward:SOLVED_REWARD);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalEpisodes = (int)(getenv('RL_TOTAL_EPISODES') ?: TOTAL_EPISODES);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    // True Online Sarsa(lambda) learns directly from MountainCar's raw -1 reward.
    $history = $runner->train(
        $totalEpisodes,
        $evalEvery,
        EVAL_EPISODES,
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
        $episodes = $la->array($history['episode']);
        $trainArt = $plt->plot($episodes, $la->array($history['trainReward']))[0];
        $evalArt = $plt->plot($episodes, $la->array($history['evalReward']))[0];
        $plt->xlabel('Training episodes');
        $plt->ylabel('Raw reward');
        $plt->legend([$trainArt, $evalArt], ['Training raw reward', 'Evaluation raw reward']);
        $plt->show(filename:__DIR__.'/../graphics/mountaincar-true-online-sarsa-lambda-history.png');
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
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
