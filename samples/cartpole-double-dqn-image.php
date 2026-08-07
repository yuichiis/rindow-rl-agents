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
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;

const SEED = 42;
const TOTAL_STEPS = 300_000;
// An 84x84x4 float32 transition takes about 221 KiB for current and next
// observations. BUFFER_SIZE=2,000 therefore needs roughly 431 MiB.
const BUFFER_SIZE = 2_000;
const BATCH_SIZE = 32;
const LEARNING_STARTS = 1_000;
const TRAIN_EVERY = 4;
const TARGET_UPDATE_INTERVAL = 1_000;
const GAMMA = 0.99;
const LEARNING_RATE = 1.0e-4;
const EPSILON_START = 1.0;
const EPSILON_END = 0.05;
const EPSILON_DECAY_STEPS = 50_000;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = 475.0;
const SOLVED_EVALUATIONS = 3;
const MODEL_FILE = __DIR__.'/../models/cartpole-double-dqn-image.weights';

const SCREEN_HEIGHT = 400;
const SCREEN_WIDTH = 600;
const IMAGE_SIZE = 84;
const FRAME_STACK = 4;
const IMAGE_SHAPE = [IMAGE_SIZE,IMAGE_SIZE,FRAME_STACK];

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new CartPoleV1($la);
$evalEnv = new CartPoleV1($la);
rlSeedSpaces($env,$evalEnv,$seed);

// Nearest-neighbour sample positions for resizing the complete 400x600 screen.
// Keeping the whole rail visible preserves the cart's absolute position.
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
    $rgb = $environment->render(mode:'rgb_array'); // [400,600,3]

    // gather() samples its first dimension. Transpose once to sample width.
    $small = $la->gather($rgb,$rowIndices);
    $small = $la->transpose($small,[1,0,2]);
    $small = $la->gather($small,$columnIndices);
    $small = $la->transpose($small,[1,0,2]); // [84,84,3]

    $small = $la->astype($small,dtype:NDArray::float32);
    $gray = $la->reduceMean($small,axis:2);
    $gray = $la->scal(1.0/255.0,$gray);

    if ($reset || !isset($frameHistory[$environment])) {
        // Repeating the first frame avoids artificial motion at episode start.
        $frameHistory[$environment] = array_fill(0,FRAME_STACK,$gray);
    } else {
        $frames = $frameHistory[$environment];
        array_shift($frames);
        $frames[] = $gray;
        $frameHistory[$environment] = $frames;
    }
    return $la->stack($frameHistory[$environment],axis:2); // [84,84,4]
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
    ddqn:true,
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
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
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
        $art = $plt->plot($la->array($history['step']),$la->array($history['evalReward']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$art],['Image Double DQN']);
        $plt->show(filename:__DIR__.'/../graphics/cartpole-double-dqn-image-history.png');
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
    $filename = $env->show(path:__DIR__.'/../graphics/cartpole-double-dqn-image-trained.gif');
    echo "filename: {$filename}\n";
}
