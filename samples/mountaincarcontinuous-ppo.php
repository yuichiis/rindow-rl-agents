<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;

const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 2048;
const BATCH_SIZE = 64;
const EPOCHS = 10;
const EVAL_EVERY = 4096;
const EVAL_EPISODES = 5;
const SOLVED_REWARD = 90.0;
const MODEL_FILE = __DIR__.'/../models/mountaincarcontinuous-ppo-raw.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$seedText = getenv('RL_SEED');
if ($seedText !== false) {
    $seed = (int)$seedText;
    $la->setSeed($seed);
    echo "Random seed: {$seed}\n";
} else {
    echo "Random seed: system default (set RL_SEED for reproducible runs)\n";
}

$env = new ContinuousMountainCarV0($la);
$evalEnv = new ContinuousMountainCarV0($la);
if ($seedText !== false) {
    $env->actionSpace()->seed($seed);
    $env->observationSpace()->seed($seed);
    $evalEnv->actionSpace()->seed($seed + 1);
    $evalEnv->observationSpace()->seed($seed + 1);
}

$actionSpace = $env->actionSpace();
$agent = new PPOAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$actionSpace->shape()[0],
    hiddenLayers:[128, 128],
    learningRate:3.0e-4,
    clipRange:0.2,
    valueLossWeight:0.5,
    entropyWeight:0.01,
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

/*
 * 成功前のGym生報酬は操作コストだけなので、谷を登るために蓄えた
 * エネルギーの差分を学習信号として加える。評価とSolved判定には
 * Runnerが別途集計するGym生報酬を使う。
 */
$rewardFunction = static function(
    NDArray $obs,
    mixed $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) : float {
    $energy = sin(3.0 * (float)$obs[0]) + 0.5 * (float)$obs[1] ** 2;
    $nextEnergy = sin(3.0 * (float)$nextObs[0]) + 0.5 * (float)$nextObs[1] ** 2;
    return $reward + 10.0 * ($nextEnergy - $energy);
};

$runner = new Runner(
    $la, $env, $evalEnv, $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:0.99,
    gaeLambda:0.95,
    solvedReward:SOLVED_REWARD,
    //rewardFunction:$rewardFunction,
    bootstrapTruncated:false,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
} else {
    $history = $runner->train(
        $totalSteps, $evalEvery, EVAL_EPISODES, bestModelFile:$modelFile
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['step']) > 0) {
        $steps = $la->array($history['step']);
        $rawArt = $plt->plot($steps, $la->array($history['evalReward']))[0];
        $shapedArt = $plt->plot($steps, $la->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawArt, $shapedArt], ['Gym raw reward', 'Shaped reward']);
        $plt->show(filename:__DIR__.'/../graphics/mountaincarcontinuous-ppo-raw-history.png');
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
    $filename = $env->show(path:__DIR__.'/../graphics/mountaincarcontinuous-ppo-raw-trained.gif');
    echo "filename: {$filename}\n";
}
