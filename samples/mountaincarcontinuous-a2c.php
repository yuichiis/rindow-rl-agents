<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\A2C\A2CAgent;
use Rindow\RL\Agents\Agent\A2C\Runner;
use Rindow\RL\Agents\Env\ContinuousMountainCar\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;

const SEED = 42;
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
const POTENTIAL_SCALE = 10.0;
const MODEL_FILE = __DIR__.'/../models/mountaincarcontinuous-a2c-shaped.weights';
const HISTORY_FILE = __DIR__.'/../graphics/mountaincarcontinuous-a2c-shaped-history.png';
const ANIMATION_FILE = __DIR__.'/../graphics/mountaincarcontinuous-a2c-shaped-animation.gif';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new ContinuousMountainCarV0($hostLa);
$evalEnv = new ContinuousMountainCarV0($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
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
    actionMin:$nn->deviceArray($actionSpace->low()),
    actionMax:$nn->deviceArray($actionSpace->high()),
    initialLogStd:log(INITIAL_ACTION_STD),
    optimizer:'adam',
    actionKernelInitializer:$actionKernelInitializer,
    activation:'relu',
);
$agent->summary();

/*
 * Potential-based reward shaping:
 *
 *     r'(s,a,s') = r(s,a,s') + gamma * Phi(s') - Phi(s)
 *
 * PhiはMountainCarの正規化した高さと速度エネルギーから作る。速度の符号を
 * 消すことで、左右どちらへ振っていても運動エネルギーの蓄積を評価する。
 * 終端ではPhi(s')=0とし、元のGym報酬が定める最適方策を変えない。
 */
$mountainCarPotential = static function(float $position, float $velocity) : float {
    // sin(3p)を概ね[0,1]へ、速度をMountainCarの上限0.07で正規化する。
    $height = 0.5 * (sin(3.0 * $position) + 1.0);
    $normalizedVelocity = $velocity / 0.07;
    $velocityEnergy = 0.5 * $normalizedVelocity ** 2;
    return POTENTIAL_SCALE * ($height + $velocityEnergy);
};

/*
 * MountainCarの生報酬は成功するまで常に-1なので、初期方策には学習信号が
 * ほとんどない。旧版で収束を確認できた式をそのまま明示的に記述する。
 * ログにはGym生報酬(EvalReward)とこの報酬(EvalShaped)の両方を表示する。
 */
$rewardFunction = static function(
    NDArray $obs,
    mixed $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) use ($nn,$mountainCarPotential) : float {
    $obs = $nn->hostArray($obs);
    $nextObs = $nn->hostArray($nextObs);
    $position = (float)$obs[0];
    $velocity = (float)$obs[1];
    $nextPosition = (float)$nextObs[0];
    $nextVelocity = (float)$nextObs[1];

    // >>>>>>>>>>>>>>>>>>>>>>>>>
    // 1.物理エネルギーの増加量を報酬とする。
    //$energy = sin(3.0 * $position) + 0.5 * $velocity ** 2;
    //$nextEnergy = sin(3.0 * $nextPosition) + 0.5 * $nextVelocity ** 2;
    //
    //$energyGain = 10.0 * ($nextEnergy - $energy);
    //$stepPenalty = -0.1;
    //$goalBonus = $terminated ? 100.0 : 0.0;
    //return $energyGain + $stepPenalty + $goalBonus;
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>

    // <<<<<<<<<<<<<<<<<<<<<<<<<<
    // 2.速度の絶対値を直接報酬にする
    $velocityReward = 10.0 * abs($nextVelocity);
    return $velocityReward - 0.1 + ($terminated ? 100.0 : 0.0);
    // <<<<<<<<<<<<<<<<<<<<<<<<<<

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>
    // 3. Potential-based shapingにする
    //$potential = $mountainCarPotential($position, $velocity);
    //$nextPotential = ($terminated || $truncated)
    //    ? 0.0
    //    : $mountainCarPotential($nextPosition, $nextVelocity);
    //return $reward + GAMMA * $nextPotential - $potential;
};

$rolloutSteps = rlEnvInt('RL_ROLLOUT_STEPS',ROLLOUT_STEPS);
$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    rolloutSteps:$rolloutSteps,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:SOLVED_REWARD,
    bootstrapTruncated:false,
    rewardFunction:$rewardFunction,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$historyFile = rlEnvString('RL_HISTORY_FILE',HISTORY_FILE);
$animationFile = rlEnvString('RL_ANIMATION_FILE',ANIMATION_FILE);
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
    }
    if (count($history['step']) > 0) {
        $art = $plt->plot(
            $hostLa->array($history['step']),$hostLa->array($history['evalReward'])
        )[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Gym evaluation reward');
        $plt->legend([$art],['A2C with shaped training reward']);
        $plt->show(
            filename:$historyFile
        );
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
        path:$animationFile
    );
    echo "filename: {$filename}\n";
}
