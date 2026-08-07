<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\SAC\SACGSDEAgent;
use Rindow\RL\Agents\Agent\SAC\Runner;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;

const SEED = 42;
const TOTAL_STEPS = 100_000;
const START_STEPS = 1_000;
const BATCH_SIZE = 256;
const BUFFER_SIZE = 100_000;
const LR_ACTOR = 3.0e-4;
const LR_CRITIC = 3.0e-4;
const LR_ALPHA = 3.0e-4;
const GAMMA = 0.99;
const TAU = 0.005;
const HIDDEN_DIM = 256;
const ALPHA_INIT = 1.0;
const GSDE_LATENT_DIM = 64;
const GSDE_RESET_FREQ = 16;
const UPDATE_EVERY = 1;
const EVAL_EVERY = 2_000;
const EVAL_EPISODES = 5;
const SOLVED_REWARD = 90.0;
const MODEL_FILE = __DIR__.'/../models/mountaincarcontinuous-sac-gsde-shaped.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed(SEED);
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new ContinuousMountainCarV0($la);
$evalEnv = new ContinuousMountainCarV0($la);
$env->observationSpace()->seed(SEED);
$env->actionSpace()->seed(SEED);
$evalEnv->observationSpace()->seed(SEED+1);
$evalEnv->actionSpace()->seed(SEED+1);

$obsDim = $env->observationSpace()->shape()[0];
$actionSpace = $env->actionSpace();
$actDim = $actionSpace->shape()[0];
$high = $actionSpace->high()->toArray();
while (is_array($high)) $high = reset($high);
$actLimit = (float)$high;

echo "gSDE latent_dim=".GSDE_LATENT_DIM." reset_freq=".GSDE_RESET_FREQ."\n";
echo "obs_dim={$obsDim} act_dim={$actDim} act_limit={$actLimit}\n";

$agent = new SACGSDEAgent(
    $nn,$obsDim,$actDim,$actLimit,GSDE_LATENT_DIM,HIDDEN_DIM,
    LR_ACTOR,LR_CRITIC,LR_ALPHA,ALPHA_INIT,GAMMA,TAU,BATCH_SIZE,
);
$agent->summary();

/*
 * ゴール前のGym生報酬は主に操作コストだけなので、位置と速度から
 * 求めたエネルギー増分を学習信号へ加える。評価とSolved判定には
 * Runnerが別途集計するGym生報酬を使う。
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
    return $reward+10.0*($nextEnergy-$energy);
};

$bufferSize = (int)(getenv('RL_BUFFER_SIZE') ?: BUFFER_SIZE);
$runner = new Runner(
    $la,$nn,$env,$evalEnv,$agent,$obsDim,$actDim,$actLimit,$bufferSize,
    solvedReward:SOLVED_REWARD,
    rewardFunction:$rewardFunction,
);

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$startSteps = (int)(getenv('RL_START_STEPS') !== false
    ? getenv('RL_START_STEPS') : START_STEPS);
$updateEvery = (int)(getenv('RL_UPDATE_EVERY') ?: UPDATE_EVERY);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);
$evalEpisodes = (int)(getenv('RL_EVAL_EPISODES') ?: EVAL_EPISODES);

if (is_file($modelFile)) {
    echo "Model weights found. Loading: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
    echo "Training skipped.\n";
} else {
    $history = $runner->train(
        $totalSteps,$startSteps,$updateEvery,GSDE_RESET_FREQ,
        $evalEvery,$evalEpisodes,evalgSDE:true,
    );

    if (count($history['step']) > 0) {
        $steps = $la->array($history['step']);
        $rawArt = $plt->plot($steps,$la->array($history['evalDet']))[0];
        $shapedArt = $plt->plot($steps,$la->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawArt,$shapedArt],['Gym raw reward','Shaped reward']);
        $plt->show(
            filename:__DIR__.'/../graphics/mountaincarcontinuous-sac-gsde-shaped-history.png'
        );
    }
    $agent->saveWeightsToFile($modelFile);
    echo "Model weights saved: {$modelFile}\n";
}

if (getenv('RL_SKIP_DEMO') !== '1') {
    echo "Creating demo animation.\n";
    for ($episode=1; $episode<=5; $episode++) {
        [$obs,$info] = $env->reset();
        $env->render();
        $done = false;
        $steps = 0;
        $rawTotal = 0.0;
        $shapedTotal = 0.0;
        while (!$done) {
            $action = $agent->selectActionDeterministic($obs);
            $currentObs = $obs;
            [$obs,$reward,$terminated,$truncated,$info] = $env->step($action);
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
        path:__DIR__.'/../graphics/mountaincarcontinuous-sac-gsde-shaped-trained.gif'
    );
    echo "filename: {$filename}\n";
}
