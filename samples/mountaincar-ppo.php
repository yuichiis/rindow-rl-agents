<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 2048;
const BATCH_SIZE = 64;
const EPOCHS = 10;
const GAMMA = 0.99;
const GAE_LAMBDA = 0.95;
const LEARNING_RATE = 3.0e-4;
const CLIP_RANGE = 0.2;
const VALUE_LOSS_WEIGHT = 0.5;
const ENTROPY_WEIGHT = 0.001;
const EVAL_EVERY = 2048;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = -110.0;
// 失敗した旧構成のcheckpointとの誤読込を避けるため別名にする。
const MODEL_FILE = __DIR__.'/../models/mountaincar-ppo-shared.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

// 旧版はseedを固定していない。比較実験で再現性が必要な場合だけ指定する。
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

$agent = new PPOAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128, 128],
    learningRate:LEARNING_RATE,
    clipRange:CLIP_RANGE,
    valueLossWeight:VALUE_LOSS_WEIGHT,
    entropyWeight:ENTROPY_WEIGHT,
    epochs:EPOCHS,
    batchSize:BATCH_SIZE,
    maxGradNorm:0.5,
    clipValueLoss:true,
    sharedBackbone:true,
);
$agent->summary();

/*
 * MountainCarの生報酬は成功するまで常に-1なので、初期方策には学習信号が
 * ほとんどない。旧版で収束を確認できた式をそのまま明示的に記述する。
 * ログにはGym生報酬(EvalReward)とこの報酬(EvalShaped)の両方を表示する。
 */
$mountainCarReward = static function(
    NDArray $obs,
    int $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) : float {
    $position = (float)$obs[0];
    $velocity = (float)$obs[1];
    $nextPosition = (float)$nextObs[0];
    $nextVelocity = (float)$nextObs[1];

    $energy = sin(3.0 * $position) + 0.5 * $velocity ** 2;
    $nextEnergy = sin(3.0 * $nextPosition) + 0.5 * $nextVelocity ** 2;

    $energyGain = 10.0 * ($nextEnergy - $energy);
    $stepPenalty = -0.1;
    $goalBonus = $terminated ? 100.0 : 0.0;
    return $energyGain + $stepPenalty + $goalBonus;
};

$runner = new Runner(
    $la,
    $env,
    $evalEnv,
    $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:SOLVED_REWARD,
    rewardFunction:$mountainCarReward,
    bootstrapTruncated:false,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,
        $evalEvery,
        EVAL_EPISODES,
        bestModelFile:$modelFile,
    );
    // デモと最終保存には、学習末尾ではなく評価が最高だった方策を使う。
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    } else {
        $agent->saveWeightsToFile($modelFile);
        echo "Model saved: {$modelFile}\n";
    }

    if (count($history['step']) > 0) {
        $steps = $la->array($history['step']);
        $rawRewardArt = $plt->plot($steps, $la->array($history['evalReward']))[0];
        $shapedRewardArt = $plt->plot($steps, $la->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawRewardArt, $shapedRewardArt], ['Gym raw reward', 'Shaped reward']);
        $plt->show(filename:__DIR__.'/../graphics/mountaincar-ppo-history.png');
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    echo "Creating demo animation.\n";
    for ($episode = 1; $episode <= 5; $episode++) {
        [$obs] = $env->reset();
        $done = false;
        $rawTotal = 0.0;
        $shapedTotal = 0.0;
        $steps = 0;
        $goal = false;
        $env->render();
        while (!$done) {
            $currentObs = $obs;
            $action = $la->array(
                $agent->selectActionDeterministic($obs),
                dtype:NDArray::int32,
            );
            [$obs, $reward, $terminated, $truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $rawTotal += $reward;
            $shapedTotal += $mountainCarReward(
                $currentObs,
                (int)$la->scalar($action),
                $obs,
                $reward,
                $terminated,
                $truncated,
            );
            $goal = $goal || $terminated;
            $steps++;
            $env->render();
        }
        printf(
            "Test Episode %d | Steps=%d | RawReward=%+.1f | ShapedReward=%+.1f | Goal=%s\n",
            $episode, $steps, $rawTotal, $shapedTotal, $goal ? 'yes' : 'no'
        );
    }
    $filename = $env->show(path:__DIR__.'/../graphics/mountaincar-ppo-trained.gif');
    echo "filename: {$filename}\n";
}
