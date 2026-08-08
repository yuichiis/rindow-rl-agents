<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\AI\RL\Environment;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;
use Rindow\RL\Agents\Agent\DQN\Runner;
use Rindow\RL\Agents\Env\CartPole\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;

const SEED = 42;
const TOTAL_STEPS = 300_000;
// Image transitions are much larger than vector observations. At 34x100x4,
// both observation arrays in this buffer use about 218 MB as float32.
const BUFFER_SIZE = 2_000;
const BATCH_SIZE = 32;
const LEARNING_STARTS = 1_000;
const TRAIN_EVERY = 4;
const TARGET_UPDATE_INTERVAL = 1_000;
const GAMMA = 0.99;
const LEARNING_RATE = 2.5e-4;
const EPSILON_START = 1.0;
const EPSILON_END = 0.05;
const EPSILON_DECAY_STEPS = 100_000;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = 475.0;
const SOLVED_EVALUATIONS = 3;
const MODEL_FILE = __DIR__.'/../models/cartpole-dqn-image.weights';

const CROP_TOP = 60;
const CROP_HEIGHT = 200;
const SCREEN_WIDTH = 600;
const DOWNSAMPLE = 6;
const FRAME_STACK = 4;
const IMAGE_HEIGHT = 34; // count(range(0, CROP_HEIGHT-1, DOWNSAMPLE))
const IMAGE_WIDTH = 100; // count(range(0, SCREEN_WIDTH-1, DOWNSAMPLE))
const IMAGE_SHAPE = [IMAGE_HEIGHT,IMAGE_WIDTH,FRAME_STACK];

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new CartPoleV1($hostLa);
$evalEnv = new CartPoleV1($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
}

$rowIndices = $hostLa->array(range(0,CROP_HEIGHT-1,DOWNSAMPLE),dtype:NDArray::int32);
$columnIndices = $hostLa->array(range(0,SCREEN_WIDTH-1,DOWNSAMPLE),dtype:NDArray::int32);
$frameHistory = new WeakMap();

// The third argument is true immediately after reset. Keeping separate histories
// for env and evalEnv prevents evaluations from contaminating training frames.
$imageObservation = static function(
    Environment $environment,
    mixed $rawObservation,
    bool $reset=false,
) use ($nn,$hostLa,$rowIndices,$columnIndices,$frameHistory) : NDArray {
    $rgb = $environment->render(mode:'rgb_array');       // [400,600,3]
    $croppedView = $hostLa->slice(
        $rgb,
        begin:[CROP_TOP,0],
        size:[CROP_HEIGHT,SCREEN_WIDTH],
    );
    // imagecopy materializes the selected image area as an independent NDArray.
    $small = $hostLa->gather($croppedView,$rowIndices);
    $small = $hostLa->transpose($small,[1,0,2]);
    $small = $hostLa->gather($small,$columnIndices);
    $small = $hostLa->transpose($small,[1,0,2]);
    $small = $hostLa->astype($small,dtype:NDArray::float32);
    $gray = $hostLa->reduceMean($small,axis:2);
    $gray = $hostLa->scal(1.0/255.0,$gray);

    if ($reset || !isset($frameHistory[$environment])) {
        $frameHistory[$environment] = array_fill(0,FRAME_STACK,$gray);
    } else {
        $frames = $frameHistory[$environment];
        array_shift($frames);
        $frames[] = $gray;
        $frameHistory[$environment] = $frames;
    }
    $frames = $hostLa->stack($frameHistory[$environment],axis:2);
    return $nn->deviceArray($frames);
};

$agent = new DQNAgent(
    $nn,
    obsDim:IMAGE_SHAPE,
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128],
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
        // Preserve the 3x3 spatial layout so the dense layer can distinguish
        // where the cart and pole appear on the rail.
        $nn->layers->Flatten(),
    ],
    learningRate:LEARNING_RATE,
    gamma:GAMMA,
    batchSize:BATCH_SIZE,
    targetUpdateInterval:TARGET_UPDATE_INTERVAL,
    maxGradNorm:10.0,
);
$agent->summary();

$bufferSize = rlEnvInt('RL_BUFFER_SIZE',BUFFER_SIZE);
$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    obsDim:IMAGE_SHAPE,
    bufferSize:$bufferSize,
    solvedReward:SOLVED_REWARD,
    solvedEvaluations:SOLVED_EVALUATIONS,
    observationFunction:$imageObservation,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
$learningStarts = rlEnvInt('RL_LEARNING_STARTS',LEARNING_STARTS);
$trainEvery = rlEnvInt('RL_TRAIN_EVERY',TRAIN_EVERY);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,$learningStarts,$trainEvery,$evalEvery,$evalEpisodes,
        EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,$modelFile
    );
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $hostLa->array($history['step']),$hostLa->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art],['Image DQN']);
        $plt->show(filename:__DIR__.'/../graphics/cartpole-dqn-image-history.png');
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
        $done = false;
        $totalReward = 0.0;
        $steps = 0;
        while (!$done) {
            $actionValue = $agent->selectActionDeterministic($observation);
            $action = $la->array($actionValue,dtype:NDArray::int32);
            [$rawObservation,$reward,$terminated,$truncated] = $env->step($action);
            $observation = $imageObservation($env,$rawObservation,false);
            $done = $terminated || $truncated;
            $totalReward += $reward;
            $steps++;
            $env->render();
        }
        echo "Test Episode {$episode}, Steps: {$steps}, Total Reward: {$totalReward}\n";
    }
    $filename = $env->show(path:__DIR__.'/../graphics/cartpole-dqn-image-trained.gif');
    echo "filename: {$filename}\n";
}
