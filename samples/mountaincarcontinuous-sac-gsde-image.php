<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\AI\RL\Environment;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\SAC\SACGSDEAgent;
use Rindow\RL\Agents\Agent\SAC\Runner;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;

const SEED = 42;
const TOTAL_STEPS = 300_000;
// Current and next float32 images use about 141 MiB at this buffer size.
const BUFFER_SIZE = 2_000;
const BATCH_SIZE = 32;
const START_STEPS = 1_000;
const UPDATE_EVERY = 4;
const HIDDEN_DIM = 128;
const GSDE_LATENT_DIM = 32;
const GSDE_RESET_FREQ = 16;
const LR_ACTOR = 1.0e-4;
const LR_CRITIC = 3.0e-4;
const LR_ALPHA = 3.0e-4;
const ALPHA_INIT = 1.0;
const GAMMA = 0.99;
const TAU = 0.005;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 5;
const SOLVED_REWARD = 90.0;
const MODEL_FILE = __DIR__.'/../models/mountaincarcontinuous-sac-gsde-image.weights';

const SCREEN_HEIGHT = 400;
const SCREEN_WIDTH = 600;
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

$env = new ContinuousMountainCarV0($la);
$evalEnv = new ContinuousMountainCarV0($la);
rlSeedSpaces($env,$evalEnv,$seed);

// Keep the whole track visible and stack four grayscale frames so position,
// velocity and movement direction can be inferred from rendered images alone.
$rowValues = [];
$columnValues = [];
for ($i=0; $i<IMAGE_SIZE; $i++) {
    $rowValues[] = (int)round($i*(SCREEN_HEIGHT-1)/(IMAGE_SIZE-1));
    $columnValues[] = (int)round($i*(SCREEN_WIDTH-1)/(IMAGE_SIZE-1));
}
$rowIndices = $la->array($rowValues,dtype:NDArray::int32);
$columnIndices = $la->array($columnValues,dtype:NDArray::int32);
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

$featureLayers = [
    $nn->layers->Conv2D(
        filters:8,
        kernel_size:5,
        strides:2,
        activation:'relu',
        input_shape:IMAGE_SHAPE,
        kernel_initializer:'he_normal',
    ),
    $nn->layers->MaxPooling2D(),
    $nn->layers->Conv2D(
        filters:16,
        kernel_size:5,
        strides:2,
        activation:'relu',
        kernel_initializer:'he_normal',
    ),
    $nn->layers->Conv2D(
        filters:16,
        kernel_size:3,
        activation:'relu',
        kernel_initializer:'he_normal',
    ),
    $nn->layers->Flatten(),
];

$agent = new SACGSDEAgent(
    $nn,
    obsDim:IMAGE_SHAPE,
    actDim:$actDim,
    actLimit:$actLimit,
    gsdeLatentDim:GSDE_LATENT_DIM,
    hiddenDim:HIDDEN_DIM,
    lrActor:LR_ACTOR,
    lrCritic:LR_CRITIC,
    lrAlpha:LR_ALPHA,
    alphaInit:ALPHA_INIT,
    gamma:GAMMA,
    tau:TAU,
    batchSize:BATCH_SIZE,
    featureLayers:$featureLayers,
);
$agent->summary();

$bufferSize = rlEnvInt('RL_BUFFER_SIZE',BUFFER_SIZE);
$runner = new Runner(
    $la,$nn,$env,$evalEnv,$agent,
    obsDim:IMAGE_SHAPE,
    actDim:$actDim,
    actLimit:$actLimit,
    bufferSize:$bufferSize,
    solvedReward:SOLVED_REWARD,
    // Intentionally no rewardFunction: train with the Gym reward only.
    observationFunction:$imageObservation,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$startSteps = rlEnvInt('RL_START_STEPS',START_STEPS);
$updateEvery = rlEnvInt('RL_UPDATE_EVERY',UPDATE_EVERY);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,$startSteps,$updateEvery,GSDE_RESET_FREQ,
        $evalEvery,$evalEpisodes,evalgSDE:false,
    );
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $la->array($history['step']),$la->array($history['evalDet'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Gym evaluation reward');
        $plt->legend([$art],['Image SAC+gSDE']);
        $plt->show(
            filename:__DIR__.'/../graphics/mountaincarcontinuous-sac-gsde-image-history.png'
        );
    }
    $agent->saveWeightsToFile($modelFile);
    echo "Model saved: {$modelFile}\n";
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
    for ($episode=1; $episode<=5; $episode++) {
        [$rawObservation] = $env->reset();
        $observation = $imageObservation($env,$rawObservation,true);
        $done = false;
        $totalReward = 0.0;
        $steps = 0;
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
            "Test Episode %d | Steps=%d | Total Reward=%+.1f | Goal=%s\n",
            $episode,$steps,$totalReward,$terminated ? 'yes' : 'no'
        );
    }
    $filename = $env->show(
        path:__DIR__.'/../graphics/mountaincarcontinuous-sac-gsde-image-trained.gif'
    );
    echo "filename: {$filename}\n";
}
