<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DDPG\DDPGAgent;
use Rindow\RL\Agents\Agent\DDPG\Runner;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;

const SEED = 42;
const TOTAL_STEPS = 100_000;
const START_STEPS = 5_000;
const UPDATE_AFTER = 1_000;
const UPDATE_EVERY = 50;
const BATCH_SIZE = 128;
const BUFFER_SIZE = 100_000;
const HIDDEN_DIM = 256;
const LR_ACTOR = 1.0e-4;
const LR_CRITIC = 1.0e-3;
const GAMMA = 0.99;
const TAU = 0.005;
const NOISE_SIGMA = 0.20;
const EVAL_EVERY = 2_000;
const EVAL_EPISODES = 5;
const MODEL_FILE = __DIR__.'/../models/pendulum-ddpg.weights';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$nn = new NeuralNetworks($mo); $plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new PendulumV1($la); $evalEnv = new PendulumV1($la);
rlSeedSpaces($env,$evalEnv,$seed);
$obsDim = $env->observationSpace()->shape()[0];
$actionSpace = $env->actionSpace(); $actDim = $actionSpace->shape()[0];
$high = $actionSpace->high()->toArray(); while (is_array($high)) $high=reset($high);
$actLimit = (float)$high;

$agent = new DDPGAgent($nn,$obsDim,$actDim,$actLimit,HIDDEN_DIM,LR_ACTOR,LR_CRITIC,GAMMA,TAU,BATCH_SIZE);
$agent->summary();
$runner = new Runner($la,$env,$evalEnv,$agent,$obsDim,$actDim,$actLimit,BUFFER_SIZE,
    solvedReward:-200.0,noiseSigma:NOISE_SIGMA);
$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
$startSteps = rlEnvInt('RL_START_STEPS',START_STEPS);
$updateAfter = rlEnvInt('RL_UPDATE_AFTER',UPDATE_AFTER);
$updateEvery = rlEnvInt('RL_UPDATE_EVERY',UPDATE_EVERY);
if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile); echo "Model loaded: {$modelFile}\n";
} else {
    $history = $runner->train($totalSteps,$startSteps,$updateAfter,$updateEvery,$evalEvery,$evalEpisodes,$modelFile);
    if (is_file($modelFile)) $agent->loadWeightsFromFile($modelFile);
    if (count($history['step'])) {
        $art=$plt->plot($la->array($history['step']),$la->array($history['evalReward']))[0];
        $plt->xlabel('Training steps'); $plt->ylabel('Evaluation reward'); $plt->legend([$art],['DDPG']);
        $plt->show(filename:__DIR__.'/../graphics/pendulum-ddpg-history.png');
    }
}
if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
    for ($episode=1;$episode<=5;$episode++) {
        [$obs]=$env->reset(); $done=false; $total=0.0; $steps=0; $env->render();
        while (!$done) {
            $action=$agent->selectActionDeterministic($obs);
            [$obs,$reward,$terminated,$truncated]=$env->step($action);
            $done=$terminated||$truncated; $total+=$reward; $steps++; $env->render();
        }
        printf("Test Episode %d, Steps: %d, Total Reward: %.1f\n",$episode,$steps,$total);
    }
    echo 'filename: '.$env->show(path:__DIR__.'/../graphics/pendulum-ddpg-trained.gif')."\n";
}
