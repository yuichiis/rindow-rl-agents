<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\A2C\A2CAgent;
use Rindow\RL\Agents\Agent\A2C\Runner;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const SEED = 42;
const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 32;
const GAMMA = 0.99;
const GAE_LAMBDA = 0.95;
const LEARNING_RATE = 3.0e-4;
const VALUE_LOSS_WEIGHT = 0.5;
const ENTROPY_WEIGHT = 0.01;
const EVAL_EVERY = 5_000;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = -110.0;
const MODEL_FILE = __DIR__.'/../models/mountaincar-a2c-shaped.weights';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";

$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new MountainCarV0($la);
$evalEnv = new MountainCarV0($la);
rlSeedSpaces($env,$evalEnv,$seed);

$agent = new A2CAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128,128],
    learningRate:LEARNING_RATE,
    valueLossWeight:VALUE_LOSS_WEIGHT,
    entropyWeight:ENTROPY_WEIGHT,
    maxGradNorm:0.5,
    normalizeAdvantages:true,
    optimizer:'adam',
    activation:'relu',
);
$agent->summary();

/*
 * MountainCarのGym生報酬は成功まで常に-1なので、位置と速度から
 * 算出したエネルギー増分を学習信号として与える。評価とSolved判定は
 * RunnerがGym生報酬で行う。
 */
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

    $energy = sin(3.0*$position) + 0.5*$velocity**2;
    $nextEnergy = sin(3.0*$nextPosition) + 0.5*$nextVelocity**2;

    $energyGain = 10.0*($nextEnergy-$energy);
    $stepPenalty = -0.1;
    $goalBonus = $terminated ? 100.0 : 0.0;
    return $energyGain+$stepPenalty+$goalBonus;
};

$rolloutSteps = rlEnvInt('RL_ROLLOUT_STEPS',ROLLOUT_STEPS);
$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    rolloutSteps:$rolloutSteps,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:SOLVED_REWARD,
    bootstrapTruncated:false,
    rewardFunction:$mountainCarReward,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);

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
    } else {
        $agent->saveWeightsToFile($modelFile);
        echo "Model saved: {$modelFile}\n";
    }
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $la->array($history['step']),$la->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Gym evaluation reward');
        $plt->legend([$art],['A2C with shaped training reward']);
        $plt->show(filename:__DIR__.'/../graphics/mountaincar-a2c-shaped-history.png');
    }
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
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
    $filename = $env->show(path:__DIR__.'/../graphics/mountaincar-a2c-shaped-trained.gif');
    echo "filename: {$filename}\n";
}
