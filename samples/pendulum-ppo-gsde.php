<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;
use Rindow\RL\Agents\Env\Pendulum\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;

const SEED = 42;
const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 1024;
const EVAL_EVERY = 1024;
const EVAL_EPISODES = 10;
const MODEL_FILE = __DIR__.'/../models/pendulum-ppo-gsde.weights';
const HISTORY_FILE = __DIR__.'/../graphics/pendulum-ppo-gsde-history.png';
const ANIMATION_FILE = __DIR__.'/../graphics/pendulum-ppo-gsde-trained.gif';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$env = new PendulumV1($hostLa);
$evalEnv = new PendulumV1($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
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
    entropyWeight:0.0,
    epochs:10,
    batchSize:64,
    maxGradNorm:0.5,
    clipValueLoss:true,
    sharedBackbone:true,
    continuous:true,
    actionMin:$nn->deviceArray($actionSpace->low()),
    actionMax:$nn->deviceArray($actionSpace->high()),
    exploration:'gsde',
    // -1 keeps one exploration matrix for each complete PPO rollout.
    sdeSampleFreq:-1,
    sdeInitialLogStd:-2.0,
);
$agent->summary();

$runner = new Runner(
    $la, $env, $evalEnv, $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:0.90,
    gaeLambda:0.95,
    solvedReward:-200.0,
    bootstrapTruncated:false,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$historyFile = rlEnvString('RL_HISTORY_FILE',HISTORY_FILE);
$animationFile = rlEnvString('RL_ANIMATION_FILE',ANIMATION_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
} else {
    $history = $runner->train($totalSteps, $evalEvery, $evalEpisodes, bestModelFile:$modelFile);
    if (is_file($modelFile)) $agent->loadWeightsFromFile($modelFile);
    if (count($history['step']) > 0) {
        $steps = $hostLa->array($history['step']);
        $rewardArt = $plt->plot($steps, $hostLa->array($history['evalReward']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rewardArt], ['PPO gSDE']);
        $plt->show(filename:$historyFile);
    }
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
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
    $filename = $env->show(path:$animationFile);
    echo "filename: {$filename}\n";
}
