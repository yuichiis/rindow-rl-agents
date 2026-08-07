<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\A2C\A2CAgent;
use Rindow\RL\Agents\Agent\A2C\Runner;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;

const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 32;
const GAMMA = 0.99;
const GAE_LAMBDA = 0.95;
const LEARNING_RATE = 3.0e-4;
const VALUE_LOSS_WEIGHT = 0.5;
const ENTROPY_WEIGHT = 0.001;
const INITIAL_ACTION_STD = 0.5;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 5;
const SOLVED_REWARD = 90.0;
const MODEL_FILE = __DIR__.'/../models/mountaincarcontinuous-a2c-shaped.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$seedText = getenv('RL_SEED');
if ($seedText !== false) {
    $seed = (int)$seedText;
    $la->setSeed($seed);
    echo "Random seed: {$seed}\n";
} else {
    echo "Random seed: system default (set RL_SEED for reproducible runs)\n";
}

$env = new ContinuousMountainCarV0($la);
$evalEnv = new ContinuousMountainCarV0($la);
if ($seedText !== false) {
    $env->actionSpace()->seed($seed);
    $env->observationSpace()->seed($seed);
    $evalEnv->actionSpace()->seed($seed+1);
    $evalEnv->observationSpace()->seed($seed+1);
}

$actionSpace = $env->actionSpace();
$actionKernelInitializer = $nn->backend()->getInitializer(
    'random_uniform',minval:-0.003,maxval:0.003
);
$agent = new A2CAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$actionSpace->shape()[0],
    hiddenLayers:[128,128],
    learningRate:LEARNING_RATE,
    valueLossWeight:VALUE_LOSS_WEIGHT,
    entropyWeight:ENTROPY_WEIGHT,
    maxGradNorm:0.5,
    normalizeAdvantages:true,
    continuous:true,
    actionMin:$actionSpace->low(),
    actionMax:$actionSpace->high(),
    initialLogStd:log(INITIAL_ACTION_STD),
    optimizer:'adam',
    actionKernelInitializer:$actionKernelInitializer,
    activation:'relu',
);
$agent->summary();

/*
 * 成功前のGym生報酬は操作コストだけなので、位置と速度から求めた
 * エネルギーの増分を学習信号へ加える。評価とSolved判定にはRunnerが
 * 別途集計するGym生報酬を使う。
 */
$rewardFunction = static function(
    NDArray $obs,
    mixed $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) : float {
    $energy = sin(3.0*(float)$obs[0]) + 0.5*(float)$obs[1]**2;
    $nextEnergy = sin(3.0*(float)$nextObs[0]) + 0.5*(float)$nextObs[1]**2;
    return $reward + 10.0*($nextEnergy-$energy);
};

$rolloutSteps = (int)(getenv('RL_ROLLOUT_STEPS') ?: ROLLOUT_STEPS);
$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    rolloutSteps:$rolloutSteps,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:SOLVED_REWARD,
    bootstrapTruncated:false,
    rewardFunction:$rewardFunction,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
$evalEpisodes = (int)(getenv('RL_EVAL_EPISODES') ?: EVAL_EPISODES);

if (is_file($modelFile)) {
    $agent->loadWeightsFromFile($modelFile);
    echo "Model loaded: {$modelFile}\n";
    printf("Evaluation reward: %.1f\n",$runner->evaluate($evalEpisodes));
} else {
    $history = $runner->train(
        $totalSteps,$evalEvery,$evalEpisodes,bestModelFile:$modelFile
    );
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    }
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $la->array($history['step']),$la->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Gym evaluation reward');
        $plt->legend([$art],['A2C with shaped training reward']);
        $plt->show(
            filename:__DIR__.'/../graphics/mountaincarcontinuous-a2c-shaped-history.png'
        );
    }
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    echo "Creating demo animation.\n";
    for ($episode=1; $episode<=5; $episode++) {
        [$obs] = $env->reset();
        $done = false;
        $rawTotal = 0.0;
        $shapedTotal = 0.0;
        $steps = 0;
        $env->render();
        while (!$done) {
            $action = $agent->selectActionDeterministic($obs);
            $currentObs = $obs;
            [$obs,$reward,$terminated,$truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $rawTotal += $reward;
            $shapedTotal += $rewardFunction(
                $currentObs,$action,$obs,$reward,$terminated,$truncated
            );
            $steps++;
            $env->render();
        }
        printf(
            "Test Episode %d | Steps=%d | RawReward=%+.1f | ShapedReward=%+.1f | Goal=%s\n",
            $episode,$steps,$rawTotal,$shapedTotal,$terminated ? 'yes' : 'no'
        );
    }
    $filename = $env->show(
        path:__DIR__.'/../graphics/mountaincarcontinuous-a2c-shaped-trained.gif'
    );
    echo "filename: {$filename}\n";
}
