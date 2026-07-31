<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;
use Rindow\RL\Agents\Agent\SAC\SACGSDEAgent;
use Rindow\RL\Agents\Agent\SAC\Runner;

# ─────────────────────────────────────────────
# ハイパーパラメータ
# ─────────────────────────────────────────────
const ENV_ID          = "MountainCarContinuous-v0";
const SEED            = 42;
const TOTAL_STEPS     = 13000; # 100000;
const START_STEPS     = 1000;
const BATCH_SIZE      = 256;
const BUFFER_SIZE     = 13000; # 100000;
const LR_ACTOR        = 3e-4;
const LR_CRITIC       = 3e-4;
const LR_ALPHA        = 3e-4;
const GAMMA           = 0.99;
const TAU             = 0.005;
const HIDDEN_DIM      = 256;
const ALPHA_INIT      = 1.0;
const GSDE_LATENT_DIM = 64;
const GSDE_RESET_FREQ = 16;
const UPDATE_EVERY    = 1;
const EVAL_EVERY      = 2_000;
const EVAL_EPISODES   = 5;
const SOLVED_REWARD   = 90.0;
const MODEL_FILE       = __DIR__ . '/../models/mountaincarcontinuous-sac-gsde.weights';

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed(SEED);
$nn = new NeuralNetworks($mo);
$plt = new Plot(['renderer.skipRunViewer' => true],$mo);

// 短縮診断用。未指定時は従来の定数を使用する。
$totalSteps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$evalEvery = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

$env = new ContinuousMountainCarV0($la);
$evalEnv = new ContinuousMountainCarV0($la);
$stateShape = $env->observationSpace()->shape();
$obsDim = $stateShape[0];
$actionSpace = $env->actionSpace();
$actDim = $actionSpace->shape()[0];
$env->observationSpace()->seed(SEED);
$env->actionSpace()->seed(SEED);
$actLimit = 1.0; 

echo "gSDE latent_dim=" . GSDE_LATENT_DIM . "  reset_freq=" . GSDE_RESET_FREQ . "\n";
echo "Env: ".ENV_ID."  obs_dim={$obsDim}  act_dim={$actDim}  act_limit={$actLimit}\n";

$agent  = new SACGSDEAgent(
    $nn,
    $obsDim,
    $actDim,
    $actLimit,
    GSDE_LATENT_DIM,
    HIDDEN_DIM,
    LR_ACTOR,
    LR_CRITIC,
    LR_ALPHA,
    ALPHA_INIT,
    GAMMA,
    TAU,
    BATCH_SIZE,
);

$agent->summary();

$modelFile = getenv('RL_MODEL_FILE') ?: MODEL_FILE;
[$obs,$info] = $env->reset(seed: SEED);

$runner = new Runner(
    $la,
    $nn,
    $env,
    $evalEnv,
    $agent,
    $obsDim,
    $actDim,
    $actLimit,
    BUFFER_SIZE,
    solvedReward: SOLVED_REWARD,
);

function fitplot($la,array $x,float $window,float $bottom) : NDArray
{
    $scale = $window/(max($x)-min($x));
    $bias = -min($x)*$scale+$bottom;
    return $la->increment($la->scal($scale,$la->array($x)),$bias);
}


if (is_file($modelFile)) {
    echo "Model weights found. Loading: {$modelFile}\n";
    $agent->loadWeightsFromFile($modelFile);
    echo "Training skipped.\n";
} else {
    $history = $runner->train(
    $totalSteps,
    START_STEPS,
    UPDATE_EVERY,
    GSDE_RESET_FREQ,
    $evalEvery,
    EVAL_EPISODES,
    evalgSDE: true,
    );

    $steps = $la->array($history['step']);
    $arts = [];
    $legend = [];
    $arts[] = $plt->plot($steps, $la->array($history['evalDet']))[0];
    $legend[] = 'EvalDet';
    $arts[] = $plt->plot($steps, $la->array($history['evalgSDE']))[0];
    $legend[] = 'EvalgSDE';
    $arts[] = $plt->plot($steps, fitplot($la, $history['alpha'], 100, 100))[0];
    $legend[] = 'Alpha';
    if (count($history['updateStep']) > 0) {
        $updateSteps = $la->array($history['updateStep']);
        $arts[] = $plt->plot($updateSteps, fitplot($la, $history['actorLoss'], 100, 100))[0];
        $arts[] = $plt->plot($updateSteps, fitplot($la, $history['criticLoss'], 100, 100))[0];
        $legend[] = 'ActorLoss';
        $legend[] = 'CriticLoss';
    }
    $plt->xlabel('Training steps');
    $plt->ylabel('Metric');
    $plt->legend($arts, $legend);
    $plt->show(filename:__DIR__.'/../graphics/mountaincarcontinuous-sac-gsde-history.png');

    $agent->saveWeightsToFile($modelFile);
    echo "Model weights saved: {$modelFile}\n";
}

echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    [$obs, $info] = $env->reset();
    $env->render();
    $done=false;
    $step = 0;
    $total = 0.0;
    while (!$done) {
        $action = $agent->selectActionDeterministic($obs);
        [$nextObs, $reward, $terminated, $truncated, $info] = $env->step($action);
        $done = $terminated || $truncated;
        $obs = $nextObs;
        $total += $reward;
        $step  += 1;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$step}, Total Reward: {$total}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'/../graphics/mountaincarcontinuous-sac-gsde-trained.gif');
echo "filename: {$filename}\n";
