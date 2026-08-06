<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Agent\Sarsa\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Agents\Agent\Sarsa\TrueOnlineSarsaLambdaAgent;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

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

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$seed = (int)(getenv('RL_SEED') ?: 1234);
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$env = new Maze(
    $la, policy:null, width:WIDTH, height:HEIGHT, exit:EXIT_STATE,
    throwInvalidAction:true, maxEpisodeSteps:MAX_EPISODE_STEPS,
);
$evalEnv = new Maze(
    $la, policy:$env->mazeRules(), width:WIDTH, height:HEIGHT, exit:EXIT_STATE,
    throwInvalidAction:true, maxEpisodeSteps:MAX_EPISODE_STEPS,
);
$env->actionSpace()->seed($seed);
$env->observationSpace()->seed($seed);
$evalEnv->actionSpace()->seed($seed + 1);
$evalEnv->observationSpace()->seed($seed + 1);

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
);
$runner = new Runner($la, $env, $evalEnv, $agent);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalEpisodes = (int)(getenv('RL_TOTAL_EPISODES') ?: TOTAL_EPISODES);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
    printf("Evaluation reward: %.1f\n", $runner->evaluate(EVAL_EPISODES));
} else {
    // Maze の生報酬 (-1 per step) をそのまま使用する。
    $history = $runner->train(
        $totalEpisodes, $evalEvery, EVAL_EPISODES, bestModelFile:$modelFile
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['episode']) > 0) {
        $episodes = $la->array($history['episode']);
        $trainArt = $plt->plot($episodes, $la->array($history['trainReward']))[0];
        $evalArt = $plt->plot($episodes, $la->array($history['evalReward']))[0];
        $plt->xlabel('Training episodes');
        $plt->ylabel('Raw reward');
        $plt->legend([$trainArt, $evalArt], ['Training reward', 'Evaluation reward']);
        $plt->show(filename:__DIR__.'/../graphics/maze-true-online-sarsa-lambda-history.png');
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
