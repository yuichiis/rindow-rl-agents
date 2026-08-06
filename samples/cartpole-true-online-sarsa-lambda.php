<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\RL\Agents\Agent\Sarsa\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Agents\Agent\Sarsa\TrueOnlineSarsaLambdaAgent;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;

const TOTAL_EPISODES = 2000;
const NUM_TILINGS = 8;
const TILES_PER_DIMENSION = 8;
const LEARNING_RATE = 0.1;
const GAMMA = 0.99;
const LAMBDA = 0.9;
const EPSILON = 0.05;
// 生報酬+1の割引無限期間価値 1/(1-GAMMA)。早期失敗を負のTD誤差にする。
const INITIAL_VALUE = 100.0;
const EVAL_EVERY = 25;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = 475.0;
const MODEL_FILE = __DIR__.'/../models/cartpole-true-online-sarsa-lambda.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$seed = (int)(getenv('RL_SEED') ?: 42);
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$env = new CartPoleV1($la);
$evalEnv = new CartPoleV1($la);
$env->observationSpace()->seed($seed);
$env->actionSpace()->seed($seed);
$evalEnv->observationSpace()->seed($seed + 1);
$evalEnv->actionSpace()->seed($seed + 1);

/*
 * CartPoleの速度と角速度の観測範囲は±INFなので、通常の運動範囲を
 * Tile Codingの有限境界として使用する。範囲外の値は端のタイルへ
 * 自動的にクリップされる。
 */
$tileCoder = new TileCoder(
    low:[-2.4, -3.0, -0.2095, -3.5],
    high:[2.4, 3.0, 0.2095, 3.5],
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
    initialValue:INITIAL_VALUE,
);
$runner = new Runner(
    $la, $env, $evalEnv, $agent, solvedReward:SOLVED_REWARD
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalEpisodes = (int)(getenv('RL_TOTAL_EPISODES') ?: TOTAL_EPISODES);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
    printf("Evaluation reward: %.1f\n", $runner->evaluate(EVAL_EPISODES));
} else {
    // CartPoleの生報酬 (+1 per step) をそのまま使用する。
    $history = $runner->train(
        $totalEpisodes, $evalEvery, EVAL_EPISODES, bestModelFile:$modelFile
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
        $plt->legend([$trainArt, $evalArt], ['Training reward', 'Evaluation reward']);
        $plt->show(filename:__DIR__.'/../graphics/cartpole-true-online-sarsa-lambda-history.png');
    }
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
        printf("Test Episode %d | Steps=%d | RawReward=%+.1f\n",
            $episode, $steps, $totalReward);
    }
    $filename = $env->show(
        path:__DIR__.'/../graphics/cartpole-true-online-sarsa-lambda-trained.gif'
    );
    echo "filename: {$filename}\n";
}
