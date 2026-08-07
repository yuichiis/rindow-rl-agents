<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\AI\RL\Environment;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\A2C\A2CAgent;
use Rindow\RL\Agents\Agent\A2C\Runner;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;

const SEED = 42;
const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 128;
const GAMMA = 0.99;
const GAE_LAMBDA = 0.95;
const LEARNING_RATE = 2.5e-4;
const VALUE_LOSS_WEIGHT = 0.5;
const ENTROPY_WEIGHT = 0.01;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = 475.0;
const MODEL_FILE = __DIR__.'/../models/cartpole-a2c-image.weights';

const SCREEN_HEIGHT = 400;
const SCREEN_WIDTH = 600;
const IMAGE_SIZE = 84;
const FRAME_STACK = 4;
const IMAGE_SHAPE = [IMAGE_SIZE,IMAGE_SIZE,FRAME_STACK];

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed(SEED);
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);
$env = new CartPoleV1($la);
$evalEnv = new CartPoleV1($la);
$env->observationSpace()->seed(SEED);
$env->actionSpace()->seed(SEED);
$evalEnv->observationSpace()->seed(SEED+1);
$evalEnv->actionSpace()->seed(SEED+1);

// Resize the complete screen with nearest-neighbour sampling so the absolute
// cart position remains visible.
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
    $small = $la->transpose($small,[1,0,2]); // [84,84,3]
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
    return $la->stack($frameHistory[$environment],axis:2); // [84,84,4]
};

$agent = new A2CAgent(
    $nn,
    obsDim:IMAGE_SHAPE,
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128],
    learningRate:LEARNING_RATE,
    valueLossWeight:VALUE_LOSS_WEIGHT,
    entropyWeight:ENTROPY_WEIGHT,
    maxGradNorm:0.5,
    normalizeAdvantages:true,
    optimizer:'adam',
    activation:'relu',
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

$rolloutSteps = (int)(getenv('RL_ROLLOUT_STEPS') ?: ROLLOUT_STEPS);
$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    rolloutSteps:$rolloutSteps,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:SOLVED_REWARD,
    observationFunction:$imageObservation,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
$evalEpisodes = (int)(getenv('RL_EVAL_EPISODES') ?: EVAL_EPISODES);

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,$evalEvery,$evalEpisodes,bestModelFile:$modelFile
    );
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $la->array($history['step']),$la->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art],['Image A2C']);
        $plt->show(filename:__DIR__.'/../graphics/cartpole-a2c-image-history.png');
    }
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model restored: {$modelFile}\n";
    } else {
        $agent->saveWeightsToFile($modelFile);
        echo "Model saved: {$modelFile}\n";
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
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
    $filename = $env->show(path:__DIR__.'/../graphics/cartpole-a2c-image-trained.gif');
    echo "filename: {$filename}\n";
}
