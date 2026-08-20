<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;
use Rindow\RL\Agents\Env\ContinuousMountainCar\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;

const SEED = 42;
const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 2048;
const BATCH_SIZE = 64;
const EPOCHS = 10;
const EVAL_EVERY = 4096;
const EVAL_EPISODES = 5;
const SOLVED_REWARD = 90.0;
const SOLVED_EVALUATIONS = 3;
const POTENTIAL_SCALE = 10.0;
const MODEL_FILE = __DIR__.'/../models/mountaincarcontinuous-ppo-gsde-raw.weights';
const HISTORY_FILE = __DIR__.'/../graphics/mountaincarcontinuous-ppo-gsde-rawhistory.png';
const ANIMATION_FILE = __DIR__.'/../graphics/mountaincarcontinuous-ppo-gsde-animation.gif';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$env = new ContinuousMountainCarV0($hostLa);
$evalEnv = new ContinuousMountainCarV0($hostLa);
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
    epochs:EPOCHS,
    batchSize:BATCH_SIZE,
    maxGradNorm:0.5,
    clipValueLoss:true,
    sharedBackbone:true,
    continuous:true,
    actionMin:$nn->deviceArray($actionSpace->low()),
    actionMax:$nn->deviceArray($actionSpace->high()),
    exploration:'gsde',
    // Reuse the interval proven with SAC+gSDE; -1 keeps one noise sample per rollout.
    sdeSampleFreq:16,
    sdeInitialLogStd:-2.0,
);
$agent->summary();

$rewardFunction = static function(
    NDArray $obs,
    mixed $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) use ($nn) : float {
    $obs = $nn->hostArray($obs);
    $nextObs = $nn->hostArray($nextObs);
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
    solvedEvaluations:SOLVED_EVALUATIONS,
    //rewardFunction:$rewardFunction,
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
    $history = $runner->train(
        $totalSteps, $evalEvery, $evalEpisodes, bestModelFile:$modelFile
    );
    if ($runner->isSolved() || !is_file($modelFile)) {
        $agent->saveWeightsToFile($modelFile);
        echo "Model saved: {$modelFile}\n";
    } else {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['step']) > 0) {
        $steps = $hostLa->array($history['step']);
        $rawArt = $plt->plot($steps, $hostLa->array($history['evalReward']))[0];
        $shapedArt = $plt->plot($steps, $hostLa->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawArt, $shapedArt], ['Gym raw reward', 'Shaped reward']);
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
        printf(
            "Test Episode %d | Steps=%d | Total Reward=%+.1f | Goal=%s\n",
            $episode, $steps, $total, $terminated ? 'yes' : 'no'
        );
    }
    $filename = $env->show(path:$animationFile);
    echo "filename: {$filename}\n";
}
