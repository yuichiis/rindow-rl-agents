<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\A2C\A2CAgent;
use Rindow\RL\Agents\Agent\A2C\Runner;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

const WIDTH = 3;
const HEIGHT = 3;
const EXIT_STATE = 8;
const MAX_EPISODE_STEPS = 100;
const TOTAL_STEPS = 50_000;
const ROLLOUT_STEPS = 32;
const EVAL_EVERY = 2_000;
const EVAL_EPISODES = 10;
const MODEL_FILE = __DIR__.'/../models/maze-a2c.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$seed = (int)(getenv('RL_SEED') ?: 1234);
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

// Evaluation must use the same maze topology as training.
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

$agent = new A2CAgent(
    $nn,
    obsDim:$env->observationSpace()['location']->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128, 128],
    learningRate:3.0e-4,
    valueLossWeight:0.5,
    entropyWeight:0.05,
    maxGradNorm:0.5,
    normalizeAdvantages:true,
    stateField:'location',
    actionMaskField:'actionMask',
);
$agent->summary();

$runner = new Runner(
    $la, $env, $evalEnv, $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:0.99,
    gaeLambda:1.0,
    bootstrapTruncated:false,
    // The environment returns -1 even on arrival. Give single-environment
    // A2C a clear terminal signal while raw evaluation still measures steps.
    rewardFunction:static fn($obs, $action, $nextObs, $reward, $terminated) =>
        $terminated ? 1.0 : -0.01,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
    printf("Evaluation reward: %.1f\n", $runner->evaluate(EVAL_EPISODES));
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
        $plt->legend([$rewardArt], ['A2C with action mask']);
        $plt->show(filename:__DIR__.'/../graphics/maze-a2c-history.png');
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
    printf("Test Episode 1, Steps: %d, Total Reward: %.1f\n", $steps, $totalReward);
    $filename = $env->show(path:__DIR__.'/../graphics/maze-a2c-trained.gif', delay:100);
    echo "filename: {$filename}\n";
}
