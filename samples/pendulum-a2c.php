<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\A2C\A2CAgent;
use Rindow\RL\Agents\Agent\A2C\Runner;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;

const TOTAL_STEPS = 300_000;
const SEED = 1234;
const ROLLOUT_STEPS = 5;
const GAMMA = 0.90;
const GAE_LAMBDA = 1.0;
const LEARNING_RATE = 7.0e-4;
const VALUE_LOSS_WEIGHT = 0.5;
const ENTROPY_WEIGHT = 0.001;
const EVAL_EVERY = 4_000;
const EVAL_EPISODES = 10;
const MODEL_FILE = __DIR__.'/../models/pendulum-a2c-gaussian-v4.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);
$seedText = getenv('RL_SEED');
$seed = $seedText === false ? SEED : (int)$seedText;
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$env = new PendulumV1($la);
$evalEnv = new PendulumV1($la);

$actionSpace = $env->actionSpace();
$actionKernelInitializer = $nn->backend()->getInitializer(
    'random_uniform', minval:-0.003, maxval:0.003
);
$agent = new A2CAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$actionSpace->shape()[0],
    hiddenLayers:[128, 128],
    learningRate:LEARNING_RATE,
    valueLossWeight:VALUE_LOSS_WEIGHT,
    entropyWeight:ENTROPY_WEIGHT,
    maxGradNorm:INF,
    normalizeAdvantages:true,
    continuous:true,
    actionMin:$actionSpace->low(),
    actionMax:$actionSpace->high(),
    initialLogStd:log(4.5),
    optimizer:'adam',
    actionKernelInitializer:$actionKernelInitializer,
    activation:'relu',
);
$agent->summary();

$runner = new Runner(
    $la, $env, $evalEnv, $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:-200.0,
    bootstrapTruncated:false,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
    printf("Evaluation reward: %.1f\n", $runner->evaluate(EVAL_EPISODES));
} else {
    $history = $runner->train($totalSteps, $evalEvery, EVAL_EPISODES, bestModelFile:$modelFile);
    if (is_file($modelFile)) $agent->loadWeightsFromFile($modelFile);
    if (count($history['step']) > 0) {
        $steps = $la->array($history['step']);
        $rewardArt = $plt->plot($steps, $la->array($history['evalReward']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rewardArt], ['A2C Gaussian']);
        $plt->show(filename:__DIR__.'/../graphics/pendulum-a2c-history.png');
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
            $action = $agent->selectActionDeterministic($observation);
            [$observation, $reward, $terminated, $truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $totalReward += $reward;
            $steps++;
            $env->render();
        }
        printf("Test Episode %d, Steps: %d, Total Reward: %.1f\n", $episode, $steps, $totalReward);
    }
    $filename = $env->show(path:__DIR__.'/../graphics/pendulum-a2c-trained.gif');
    echo "filename: {$filename}\n";
}
