<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;
use Rindow\RL\Agents\Agent\PPO\Runner;
use Rindow\RL\Agents\Env\MountainCar\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const SEED = 42;
const TOTAL_STEPS = 300_000;
const ROLLOUT_STEPS = 2048;
const BATCH_SIZE = 64;
const EPOCHS = 10; #5;
const GAMMA = 0.99;
const GAE_LAMBDA = 0.95;
const LEARNING_RATE = 3.0e-4; #1.0e-4; 
const CLIP_RANGE = 0.2;
const VALUE_LOSS_WEIGHT = 0.5; #0.25;
const ENTROPY_WEIGHT = 0.01; #0.02;0.05; #0.001;
const EVAL_EVERY = 2048;
const EVAL_EPISODES = 10;
const SOLVED_REWARD = -110.0;
//const POTENTIAL_SCALE = 10.0;
// Use a distinct name to avoid loading an incompatible checkpoint from the old setup.
const MODEL_FILE = __DIR__.'/../models/mountaincar-ppo-shaped.weights';
const HISTORY_FILE = __DIR__.'/../graphics/mountaincar-ppo-shaped-history.png';
const ANIMATION_FILE = __DIR__.'/../graphics/mountaincar-ppo-shaped-animation.gif';

$seed = rlEnvInt('RL_SEED',SEED);
$epochs = rlEnvInt('RL_EPOCHS',EPOCHS);
$learningRate = rlEnvFloat('RL_LEARNING_RATE',LEARNING_RATE);
$entropyWeight = rlEnvFloat('RL_ENTROPY_WEIGHT',ENTROPY_WEIGHT);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";
echo "Rollout steps: ".ROLLOUT_STEPS."\n";
echo "Batch size: ".BATCH_SIZE."\n";
echo "Epochs: ". $epochs . "\n";
echo "Gamma: ".GAMMA."\n";
echo "GAE lambda: ".GAE_LAMBDA."\n";
echo "Learning rate: ". $learningRate . "\n";
echo "Clip range: ".CLIP_RANGE."\n";
echo "Value loss weight: ".VALUE_LOSS_WEIGHT."\n";
echo "Entropy weight: ". $entropyWeight . "\n";

$plt = new Plot(['renderer.skipRunViewer'=>true], $mo);

$env = new MountainCarV0($hostLa);
$evalEnv = new MountainCarV0($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
}

$agent = new PPOAgent(
    $nn,
    obsDim:$env->observationSpace()->shape()[0],
    numActions:$env->actionSpace()->n(),
    hiddenLayers:[128, 128],
    learningRate:$learningRate,
    clipRange:CLIP_RANGE,
    valueLossWeight:VALUE_LOSS_WEIGHT,
    entropyWeight:$entropyWeight,
    epochs:$epochs,
    batchSize:BATCH_SIZE,
    maxGradNorm:0.5,
    clipValueLoss:true,
    sharedBackbone:true,
);
$agent->summary();

/*
 * Potential-based reward shaping:
 *
 *     r'(s,a,s') = r(s,a,s') + gamma * Phi(s') - Phi(s)
 *
 * Phi combines normalized height and kinetic energy. Squaring velocity rewards
 * energy accumulation in either direction. Setting Phi(s') to zero at a true
 * terminal state preserves the optimal policy induced by the Gym reward.
 */
// $mountainCarPotential = static function(float $position, float $velocity) : float {
//     // Map sin(3p) approximately to [0,1] and normalize by the 0.07 speed limit.
//     $height = 0.5 * (sin(3.0 * $position) + 1.0);
//     $normalizedVelocity = $velocity / 0.07;
//     $velocityEnergy = 0.5 * $normalizedVelocity ** 2;
//     return POTENTIAL_SCALE * ($height + $velocityEnergy);
// };

/*
 * MountainCar returns -1 until success, which gives an initial policy little
 * directional signal. Keep the known convergent shaping formulas explicit and
 * log both the Gym reward (EvalReward) and shaped reward (EvalShaped).
 */
$mountainCarReward = static function(
    NDArray $obs,
    int $action,
    NDArray $nextObs,
    float $reward,
    bool $terminated,
    bool $truncated,
) use ($nn) : float {
    $obs = $nn->hostArray($obs);
    $nextObs = $nn->hostArray($nextObs);
    $position = (float)$obs[0];
    $velocity = (float)$obs[1];
    $nextPosition = (float)$nextObs[0];
    $nextVelocity = (float)$nextObs[1];

    // >>>>>>>>>>>>>>>>>>>>>>>>>
    // Option 1: reward the increase in mechanical energy.
    //$energy = sin(3.0 * $position) + 0.5 * $velocity ** 2;
    //$nextEnergy = sin(3.0 * $nextPosition) + 0.5 * $nextVelocity ** 2;
    //
    //$energyGain = 10.0 * ($nextEnergy - $energy);
    //$stepPenalty = -0.1;
    //$goalBonus = $terminated ? 100.0 : 0.0;
    //return $energyGain + $stepPenalty + $goalBonus;
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>

    // <<<<<<<<<<<<<<<<<<<<<<<<<<
    // Option 2: reward absolute velocity directly.
    $velocityReward = 10.0 * abs($nextVelocity);
    return $velocityReward - 0.1 + ($terminated ? 100.0 : 0.0);
    // <<<<<<<<<<<<<<<<<<<<<<<<<<

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>
    // Option 3: potential-based shaping, which preserves the optimal policy.
    // $potential = $mountainCarPotential($position, $velocity);
    // $nextPotential = ($terminated || $truncated)
    //     ? 0.0
    //     : $mountainCarPotential($nextPosition, $nextVelocity);
    // return $reward + GAMMA * $nextPotential - $potential;
};

$runner = new Runner(
    $la,
    $env,
    $evalEnv,
    $agent,
    rolloutSteps:ROLLOUT_STEPS,
    gamma:GAMMA,
    gaeLambda:GAE_LAMBDA,
    solvedReward:SOLVED_REWARD,
    rewardFunction:$mountainCarReward,
    bootstrapTruncated:false,
);

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$historyFile = rlEnvString('RL_HISTORY_FILE',HISTORY_FILE);
$animationFile = rlEnvString('RL_ANIMATION_FILE',ANIMATION_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);
echo "Total steps: {$totalSteps}\n";
echo "Evaluation every: {$evalEvery}\n";
echo "Eval episodes: {$evalEpisodes}\n";

if (is_file($modelFile)) {
    echo "Loading model: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
} else {
    $history = $runner->train(
        $totalSteps,
        $evalEvery,
        $evalEpisodes,
        bestModelFile:$modelFile,
    );
    // Use the best evaluated policy, rather than the final training state, for demo and saving.
    if (is_file($modelFile)) {
        $agent->loadWeightsFromFile($modelFile);
        echo "Best model loaded: {$modelFile}\n";
    } else {
        $agent->saveWeightsToFile($modelFile);
        echo "Model saved: {$modelFile}\n";
    }

    if (count($history['step']) > 0) {
        $steps = $hostLa->array($history['step']);
        $rawRewardArt = $plt->plot($steps, $hostLa->array($history['evalReward']))[0];
        $shapedRewardArt = $plt->plot($steps, $hostLa->array($history['evalShaped']))[0];
        $plt->xlabel('Training steps');
        $plt->ylabel('Evaluation reward');
        $plt->legend([$rawRewardArt, $shapedRewardArt], ['Gym raw reward', 'Shaped reward']);
        $plt->show(filename:$historyFile);
    }
}

if (!rlEnvBool('RL_SKIP_DEMO')) {
    echo "Creating demo animation.\n";
    for ($episode = 1; $episode <= 5; $episode++) {
        [$obs] = $env->reset();
        $done = false;
        $rawTotal = 0.0;
        $shapedTotal = 0.0;
        $steps = 0;
        $goal = false;
        $env->render();
        while (!$done) {
            $currentObs = $obs;
            $action = $la->array(
                $agent->selectActionDeterministic($obs),
                dtype:NDArray::int32,
            );
            [$obs, $reward, $terminated, $truncated] = $env->step($action);
            $done = $terminated || $truncated;
            $rawTotal += $reward;
            $shapedTotal += $mountainCarReward(
                $currentObs,
                (int)$la->scalar($action),
                $obs,
                $reward,
                $terminated,
                $truncated,
            );
            $goal = $goal || $terminated;
            $steps++;
            $env->render();
        }
        printf(
            "Test Episode %d | Steps=%d | RawReward=%+.1f | ShapedReward=%+.1f | Goal=%s\n",
            $episode, $steps, $rawTotal, $shapedTotal, $goal ? 'yes' : 'no'
        );
    }
    $filename = $env->show(path:$animationFile);
    echo "filename: {$filename}\n";
}
