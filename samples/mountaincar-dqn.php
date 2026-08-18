<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;
use Rindow\RL\Agents\Agent\DQN\Runner;
use Rindow\RL\Agents\Env\MountainCar\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const SEED = 42;
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
const POTENTIAL_SCALE = 10.0;
const MODEL_FILE = __DIR__.'/../models/mountaincar-dqn-shaped.weights';
const HISTORY_FILE = __DIR__.'/../graphics/mountaincar-dqn-shaped-history.png';
const ANIMATION_FILE = __DIR__.'/../graphics/mountaincar-dqn-shaped-animation.gif';

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer'=>true],$mo);

$env = new MountainCarV0($hostLa);
$evalEnv = new MountainCarV0($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
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
    ddqn:true,
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
$mountainCarReward = static function(
    NDArray $obs,
    int $action,
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
    //$velocityReward = 10.0 * abs($nextVelocity);
    //return $velocityReward - 0.1 + ($terminated ? 100.0 : 0.0);
    // <<<<<<<<<<<<<<<<<<<<<<<<<<

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>
    // 3. Potential-based shapingにする
    $potential = $mountainCarPotential($position, $velocity);
    $nextPotential = ($terminated || $truncated)
        ? 0.0
        : $mountainCarPotential($nextPosition, $nextVelocity);
    return $reward + GAMMA * $nextPotential - $potential;
};


$runner = new Runner(
    $la,$env,$evalEnv,$agent,
    obsDim:$env->observationSpace()->shape()[0],
    bufferSize:BUFFER_SIZE,
    solvedReward:SOLVED_REWARD,
    solvedEvaluations:SOLVED_EVALUATIONS,
    rewardFunction:$mountainCarReward,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$historyFile = rlEnvString('RL_HISTORY_FILE',HISTORY_FILE);
$animationFile = rlEnvString('RL_ANIMATION_FILE',ANIMATION_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,LEARNING_STARTS,TRAIN_EVERY,$evalEvery,$evalEpisodes,
        EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,$modelFile
    );
    if (is_file($modelFile)) $agent->loadWeightsFromFile($modelFile);
    else $agent->saveWeightsToFile($modelFile);
    if (count($history['step']) > 0) {
        $steps = $hostLa->array($history['step']);
        $rawArt = $plt->plot($steps,$hostLa->array($history['evalReward']))[0];
        $shapedArt = $plt->plot($steps,$hostLa->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawArt,$shapedArt],['Gym raw reward','Shaped reward']);
        $plt->show(filename:$historyFile);
    }
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
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
    $filename = $env->show(path:$animationFile);
    echo "filename: {$filename}\n";
}
