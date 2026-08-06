<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;
use Rindow\RL\Agents\Agent\DQN\Runner;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const TOTAL_STEPS = 300_000;
const BUFFER_SIZE = 100_000;
const BATCH_SIZE = 64;
const LEARNING_STARTS = 1_000;
const TRAIN_EVERY = 4;
const TARGET_UPDATE_INTERVAL = 500;
const GAMMA = 0.99;
const LEARNING_RATE = 5.0e-4;
const EPSILON_START = 1.0;
const EPSILON_END = 0.02;
const EPSILON_DECAY_STEPS = 100_000;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = -110.0;
const SOLVED_EVALUATIONS = 3;
const MODEL_FILE = __DIR__.'/../models/mountaincar-dqn-shaped.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$seedText = getenv('RL_SEED');
$seed = $seedText === false ? null : (int)$seedText;
if ($seed !== null) $la->setSeed($seed);

$env = new MountainCarV0($la);
$evalEnv = new MountainCarV0($la);
if ($seed !== null) {
    $env->observationSpace()->seed($seed);
    $env->actionSpace()->seed($seed);
    $evalEnv->observationSpace()->seed($seed+1);
    $evalEnv->actionSpace()->seed($seed+1);
}

$agent = new DQNAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128,128],
    learningRate:LEARNING_RATE,
    gamma:GAMMA,
    batchSize:BATCH_SIZE,
    targetUpdateInterval:TARGET_UPDATE_INTERVAL,
    maxGradNorm:10.0,
);
$agent->summary();

$mountainCarReward = static function(
    NDArray $obs,
    int $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) : float {
    $position = (float)$obs[0];
    $velocity = (float)$obs[1];
    $nextPosition = (float)$nextObs[0];
    $nextVelocity = (float)$nextObs[1];
    $energy = sin(3.0*$position)+0.5*$velocity**2;
    $nextEnergy = sin(3.0*$nextPosition)+0.5*$nextVelocity**2;
    return 10.0*($nextEnergy-$energy)-0.1+($terminated ? 100.0 : 0.0);
};

$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    obsDim:$env->observationSpace()->shape()[0],
    bufferSize:BUFFER_SIZE,
    solvedReward:SOLVED_REWARD,
    solvedEvaluations:SOLVED_EVALUATIONS,
    rewardFunction:$mountainCarReward,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,LEARNING_STARTS,TRAIN_EVERY,$evalEvery,EVAL_EPISODES,
        EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,$modelFile
    );
    if (is_file($modelFile)) $agent->loadWeightsFromFile($modelFile);
    else $agent->saveWeightsToFile($modelFile);
    if (count($history['step']) > 0) {
        $steps = $la->array($history['step']);
        $rawArt = $plt->plot($steps,$la->array($history['evalReward']))[0];
        $shapedArt = $plt->plot($steps,$la->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawArt,$shapedArt],['Gym raw reward','Shaped reward']);
        $plt->show(filename:__DIR__.'/../graphics/mountaincar-dqn-history.png');
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    for ($episode=1; $episode<=5; $episode++) {
        [$obs] = $env->reset();
        $done = false;
        $rawTotal = 0.0;
        $shapedTotal = 0.0;
        $steps = 0;
        $goal = false;
        $env->render();
        while (!$done) {
            $currentObs = $obs;
            $actionValue = $agent->selectActionDeterministic($obs);
            $action = $la->array($actionValue,dtype:NDArray::int32);
            [$obs,$reward,$terminated,$truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $rawTotal += $reward;
            $shapedTotal += $mountainCarReward(
                $currentObs,$actionValue,$obs,$reward,$terminated,$truncated
            );
            $goal = $goal || $terminated;
            $steps++;
            $env->render();
        }
        printf(
            "Test Episode %d | Steps=%d | RawReward=%+.1f | ShapedReward=%+.1f | Goal=%s\n",
            $episode,$steps,$rawTotal,$shapedTotal,$goal ? 'yes' : 'no'
        );
    }
    $filename = $env->show(path:__DIR__.'/../graphics/mountaincar-dqn-trained.gif');
    echo "filename: {$filename}\n";
}
