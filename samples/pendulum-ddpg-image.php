<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\AI\RL\Environment;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DDPG\DDPGAgent;
use Rindow\RL\Agents\Agent\DDPG\Runner;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;

const SEED = 42;
const TOTAL_STEPS = 300_000;
// Two float32 image arrays in the replay buffer use about 147 MB.
const BUFFER_SIZE = 2_000;
const BATCH_SIZE = 32;
const START_STEPS = 1_000;
const UPDATE_AFTER = 1_000;
const UPDATE_EVERY = 4;
const HIDDEN_DIM = 128;
const LR_ACTOR = 1.0e-4;
const LR_CRITIC = 1.0e-3;
const GAMMA = 0.99;
const TAU = 0.005;
const NOISE_SIGMA = 0.20;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 5;
const SOLVED_REWARD = -200.0;
const MODEL_FILE = __DIR__.'/../models/pendulum-ddpg-image.weights';

const SCREEN_SIZE = 500;
const IMAGE_SIZE = 48;
const FRAME_STACK = 4;
const IMAGE_SHAPE = [IMAGE_SIZE,IMAGE_SIZE,FRAME_STACK];

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new PendulumV1($la);
$evalEnv = new PendulumV1($la);
rlSeedSpaces($env,$evalEnv,$seed);

// Resize the whole rendered screen and stack frames so the policy can infer
// angular velocity as well as the pendulum angle.
$indices = [];
for ($i=0; $i<IMAGE_SIZE; $i++) {
    $indices[] = (int)round($i*(SCREEN_SIZE-1)/(IMAGE_SIZE-1));
}
$rowIndices = $la->array($indices,dtype:NDArray::int32);
$columnIndices = $la->array($indices,dtype:NDArray::int32);
$frameHistory = new WeakMap();

$imageObservation = static function(
    Environment $environment,
    mixed $rawObservation,
    bool $reset=false,
) use ($la,$rowIndices,$columnIndices,$frameHistory) : NDArray {
    $rgb = $environment->render(mode:'rgb_array');
    $small = $la->gather($rgb,$rowIndices);
    $small = $la->transpose($small,[1,0,2]);
    $small = $la->gather($small,$columnIndices);
    $small = $la->transpose($small,[1,0,2]);
    $small = $la->astype($small,dtype:NDArray::float32);
    $gray = $la->scal(1.0/255.0,$la->reduceMean($small,axis:2));

    if ($reset || !isset($frameHistory[$environment])) {
        $frameHistory[$environment] = array_fill(0,FRAME_STACK,$gray);
    } else {
        $frames = $frameHistory[$environment];
        array_shift($frames);
        $frames[] = $gray;
        $frameHistory[$environment] = $frames;
    }
    return $la->stack($frameHistory[$environment],axis:2);
};

$actionSpace = $env->actionSpace();
$actDim = $actionSpace->shape()[0];
$high = $actionSpace->high()->toArray();
while (is_array($high)) $high = reset($high);
$actLimit = (float)$high;
$agent = new DDPGAgent(
    $nn,
    obsDim:IMAGE_SHAPE,
    actDim:$actDim,
    actLimit:$actLimit,
    hiddenDim:HIDDEN_DIM,
    lrActor:LR_ACTOR,
    lrCritic:LR_CRITIC,
    gamma:GAMMA,
    tau:TAU,
    batchSize:BATCH_SIZE,
    featureLayers:[
        $nn->layers->Conv2D(
            filters:16,
            kernel_size:5,
            strides:2,
            activation:'relu',
            input_shape:IMAGE_SHAPE,
            kernel_initializer:'he_normal',
        ),
        $nn->layers->MaxPooling2D(),
        $nn->layers->Conv2D(
            filters:32,
            kernel_size:5,
            strides:2,
            activation:'relu',
            kernel_initializer:'he_normal',
        ),
        $nn->layers->Conv2D(
            filters:32,
            kernel_size:3,
            strides:2,
            activation:'relu',
            kernel_initializer:'he_normal',
        ),
        $nn->layers->Flatten(),
    ],
);
$agent->summary();

$bufferSize = rlEnvInt('RL_BUFFER_SIZE',BUFFER_SIZE);
$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    obsDim:IMAGE_SHAPE,
    actDim:$actDim,
    actLimit:$actLimit,
    bufferSize:$bufferSize,
    solvedReward:SOLVED_REWARD,
    noiseSigma:NOISE_SIGMA,
    observationFunction:$imageObservation,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$startSteps = rlEnvInt('RL_START_STEPS',START_STEPS);
$updateAfter = rlEnvInt('RL_UPDATE_AFTER',UPDATE_AFTER);
$updateEvery = rlEnvInt('RL_UPDATE_EVERY',UPDATE_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,$startSteps,$updateAfter,$updateEvery,
        $evalEvery,$evalEpisodes,$modelFile
    );
    if (count($history['step'])) {
        $art = $plt->plot(
            $la->array($history['step']),$la->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art],['Image DDPG']);
        $plt->show(filename:__DIR__.'/../graphics/pendulum-ddpg-image-history.png');
    }
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model restored: {$modelFile}\n";
    } else {
        $agent->saveWeightsToFile($modelFile);
        echo "Model saved: {$modelFile}\n";
    }
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
    for ($episode=1; $episode<=5; $episode++) {
        [$rawObservation] = $env->reset();
        $observation = $imageObservation($env,$rawObservation,true);
        $done = false; $totalReward = 0.0; $steps = 0;
        while (!$done) {
            $action = $agent->selectActionDeterministic($observation);
            [$rawObservation,$reward,$terminated,$truncated] = $env->step($action);
            $observation = $imageObservation($env,$rawObservation,false);
            $done = $terminated || $truncated;
            $totalReward += $reward;
            $steps++;
            $env->render();
        }
        printf(
            "Test Episode %d, Steps: %d, Total Reward: %.1f\n",
            $episode,$steps,$totalReward
        );
    }
    $filename = $env->show(path:__DIR__.'/../graphics/pendulum-ddpg-image-trained.gif');
    echo "filename: {$filename}\n";
}
