<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

const WIDTH = 3;
const HEIGHT = 3;
const EXIT_STATE = 8;
const MAX_EPISODE_STEPS = 100;
const TOTAL_STEPS = 50_000;
const ROLLOUT_STEPS = 2048;
const EVAL_EVERY = 2048;
const EVAL_EPISODES = 10;
const MODEL_FILE = __DIR__.'/../models/maze-ppo.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$seed = (int)(getenv('RL_SEED') ?: 1234);
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

// evalEnvにも同じ迷路規則を渡し、異なる迷路を評価しないようにする。
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

$agent = new PPOAgent(
    $nn,
    obsDim:$env->observationSpace()['location']->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128, 128],
    learningRate:3.0e-4,
    clipRange:0.2,
    valueLossWeight:0.5,
    entropyWeight:0.001,
    epochs:10,
    batchSize:64,
    maxGradNorm:0.5,
    clipValueLoss:true,
    sharedBackbone:true,
    stateField:'location',
    actionMaskField:'actionMask',
);
$agent->summary();

$runner = new Runner(
    $la, $env, $evalEnv, $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:0.99,
    gaeLambda:0.95,
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
        $rewardArt = $plt->plot($steps, $la->array($history['evalReward']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rewardArt], ['PPO with action mask']);
        $plt->show(filename:__DIR__.'/../graphics/maze-ppo-history.png');
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    echo "Creating demo animation.\n";
    [$obs] = $env->reset();
    $env->render();
    $done = false;
    $totalReward = 0.0;
    $steps = 0;
    while (!$done) {
        $action = $la->array(
            $agent->selectActionDeterministic($obs), dtype:NDArray::int32
        );
        [$obs, $reward, $terminated, $truncated] = $env->step($action);
        $done = $terminated || $truncated;
        $totalReward += $reward;
        $steps++;
        $env->render();
    }
    printf("Test Episode 1, Steps: %d, Total Reward: %.1f\n", $steps, $totalReward);
    $filename = $env->show(path:__DIR__.'/../graphics/maze-ppo-trained.gif', delay:100);
    echo "filename: {$filename}\n";
}
