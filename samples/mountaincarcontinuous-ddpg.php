<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DDPG\DDPGAgent;
use Rindow\RL\Agents\Agent\DDPG\Runner;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;

const TOTAL_STEPS = 300_000;
const START_STEPS = 10_000;
const UPDATE_AFTER = 1_000;
const UPDATE_EVERY = 50;
const BATCH_SIZE = 128;
const BUFFER_SIZE = 100_000;
const HIDDEN_DIM = 256;
const LR_ACTOR = 1.0e-4;
const LR_CRITIC = 1.0e-3;
const GAMMA = 0.99;
const TAU = 0.005;
const NOISE_SIGMA = 0.30;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 5;
const SOLVED_REWARD = 90.0;
const MODEL_FILE = __DIR__.'/../models/mountaincarcontinuous-ddpg-shaped.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

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
    $evalEnv->actionSpace()->seed($seed+1);
    $evalEnv->observationSpace()->seed($seed+1);
}
$obsDim = $env->observationSpace()->shape()[0];
$actionSpace = $env->actionSpace();
$actDim = $actionSpace->shape()[0];
$high = $actionSpace->high()->toArray();
while (is_array($high)) $high = reset($high);
$actLimit = (float)$high;

$agent = new DDPGAgent(
    $nn,$obsDim,$actDim,$actLimit,HIDDEN_DIM,LR_ACTOR,LR_CRITIC,GAMMA,TAU,BATCH_SIZE
);
$agent->summary();

/*
 * ゴール前の Gym 生報酬は主に操作コストだけなので、位置と速度から
 * 求めたエネルギーの増分を学習信号に加える。評価と Solved 判定には
 * Runner が別途集計する Gym 生報酬を使う。
 */
$rewardFunction = static function(
    NDArray $obs,
    mixed $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) : float {
    $energy = sin(3.0*(float)$obs[0]) + 0.5*(float)$obs[1]**2;
    $nextEnergy = sin(3.0*(float)$nextObs[0]) + 0.5*(float)$nextObs[1]**2;
    return $reward + 10.0*($nextEnergy-$energy);
};

$runner = new Runner(
    $la,$env,$evalEnv,$agent,$obsDim,$actDim,$actLimit,BUFFER_SIZE,
    solvedReward:SOLVED_REWARD,
    noiseSigma:NOISE_SIGMA,
    rewardFunction:$rewardFunction,
);
$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
$startSteps = (int)(getenv('RL_START_STEPS') !== false ? getenv('RL_START_STEPS') : START_STEPS);
$updateAfter = (int)(getenv('RL_UPDATE_AFTER') !== false ? getenv('RL_UPDATE_AFTER') : UPDATE_AFTER);
$updateEvery = (int)(getenv('RL_UPDATE_EVERY') ?: UPDATE_EVERY);
if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
} else {
    $history = $runner->train(
        $totalSteps,$startSteps,$updateAfter,$updateEvery,$evalEvery,EVAL_EPISODES,$modelFile
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['step'])) {
        $steps = $la->array($history['step']);
        $rawArt = $plt->plot($steps,$la->array($history['evalReward']))[0];
        $shapedArt = $plt->plot($steps,$la->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawArt,$shapedArt],['Gym raw reward','Shaped reward']);
        $plt->show(filename:__DIR__.'/../graphics/mountaincarcontinuous-ddpg-shaped-history.png');
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    echo "Creating demo animation.\n";
    for ($episode=1; $episode<=5; $episode++) {
        [$obs] = $env->reset();
        $done = false; $rawTotal = 0.0; $shapedTotal = 0.0; $steps = 0;
        $env->render();
        while (!$done) {
            $action = $agent->selectActionDeterministic($obs);
            $currentObs = $obs;
            [$obs,$reward,$terminated,$truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $rawTotal += $reward;
            $shapedTotal += $rewardFunction(
                $currentObs,$action,$obs,$reward,$terminated,$truncated
            );
            $steps++;
            $env->render();
        }
        printf(
            "Test Episode %d | Steps=%d | RawReward=%+.1f | ShapedReward=%+.1f | Goal=%s\n",
            $episode,$steps,$rawTotal,$shapedTotal,$terminated ? 'yes' : 'no'
        );
    }
    $filename = $env->show(
        path:__DIR__.'/../graphics/mountaincarcontinuous-ddpg-shaped-trained.gif'
    );
    echo "filename: {$filename}\n";
}
