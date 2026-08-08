<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Agent\Sarsa\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Agents\Agent\Sarsa\TrueOnlineSarsaLambdaAgent;
use Rindow\RL\Agents\Env\Maze\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

const SEED = 1234;
const WIDTH = 3;
const HEIGHT = 3;
const EXIT_STATE = 8;
const MAX_EPISODE_STEPS = 100;
const TOTAL_EPISODES = 500;
const NUM_TILINGS = 4;
const TILES_PER_DIMENSION = 2;
const LEARNING_RATE = 0.2;
const GAMMA = 1.0;
const LAMBDA = 0.9;
const EPSILON = 0.1;
const EVAL_EVERY = 10;
const EVAL_EPISODES = 10;
const MODEL_FILE = __DIR__.'/../models/maze-true-online-sarsa-lambda.weights';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$env = new Maze(
    $hostLa, policy:null, width:WIDTH, height:HEIGHT, exit:EXIT_STATE,
    throwInvalidAction:true, maxEpisodeSteps:MAX_EPISODE_STEPS,
);
$evalEnv = new Maze(
    $hostLa, policy:$env->mazeRules(), width:WIDTH, height:HEIGHT, exit:EXIT_STATE,
    throwInvalidAction:true, maxEpisodeSteps:MAX_EPISODE_STEPS,
);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
}

$tileCoder = new TileCoder(
    low:[0.0, 0.0],
    high:[HEIGHT - 1.0, WIDTH - 1.0],
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
    stateField:'location',
    actionMaskField:'actionMask',
    nn:$nn,
);
$runner = new Runner($la, $env, $evalEnv, $agent);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalEpisodes = rlEnvInt('RL_TOTAL_EPISODES',TOTAL_EPISODES);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);

if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
    printf("Evaluation reward: %.1f\n", $runner->evaluate($evalEpisodes));
} else {
    // Maze の生報酬 (-1 per step) をそのまま使用する。
    $history = $runner->train(
        $totalEpisodes, $evalEvery, $evalEpisodes, bestModelFile:$modelFile
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['episode']) > 0) {
        $episodes = $hostLa->array($history['episode']);
        $trainArt = $plt->plot($episodes, $hostLa->array($history['trainReward']))[0];
        $evalArt = $plt->plot($episodes, $hostLa->array($history['evalReward']))[0];
        $plt->xlabel('Training episodes');
        $plt->ylabel('Raw reward');
        $plt->legend([$trainArt, $evalArt], ['Training reward', 'Evaluation reward']);
        $plt->show(filename:__DIR__.'/../graphics/maze-true-online-sarsa-lambda-history.png');
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
            $agent->selectActionDeterministic($observation), dtype:NDArray::int32
        );
        [$observation, $reward, $terminated, $truncated] = $env->step($action);
        $done = $terminated || $truncated;
        $totalReward += $reward;
        $steps++;
        $env->render();
    }
    printf("Test Episode 1 | Steps=%d | RawReward=%+.1f\n", $steps, $totalReward);
    $filename = $env->show(
        path:__DIR__.'/../graphics/maze-true-online-sarsa-lambda-trained.gif', delay:100
    );
    echo "filename: {$filename}\n";
}
