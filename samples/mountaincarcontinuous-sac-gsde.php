<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0;
use Rindow\RL\Agents\Agent\SAC\SACGSDEAgent;
use Rindow\RL\Agents\Agent\SAC\Runner;
use Rindow\RL\Agents\Env\ContinuousMountainCar\DeviceWrapper;

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

$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$plt = new Plot(['renderer.skipRunViewer' => true],$mo);
$totalSteps = rlEnvInt('RL_TOTAL_STEPS',TOTAL_STEPS);
$evalEvery = rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY);

$env = new ContinuousMountainCarV0($hostLa);
$evalEnv = new ContinuousMountainCarV0($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if ($la->accelerated()) {
    $env = new DeviceWrapper($nn,$env);
    $evalEnv = new DeviceWrapper($nn,$evalEnv);
}
$stateShape = $env->observationSpace()->shape();
$obsDim = $stateShape[0];
$actionSpace = $env->actionSpace();
$actDim = $actionSpace->shape()[0];
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

$modelFile = rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
[$obs,$info] = $env->reset(seed:$seed);

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

function fitplot($hostLa,array $x,float $window,float $bottom) : NDArray
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
    $evalEpisodes,
    evalgSDE: true,
    );

    $steps = $hostLa->array($history['step']);
    $arts = [];
    $legend = [];
    $arts[] = $plt->plot($steps, $hostLa->array($history['evalDet']))[0];
    $legend[] = 'EvalDet';
    $arts[] = $plt->plot($steps, $hostLa->array($history['evalgSDE']))[0];
    $legend[] = 'EvalgSDE';
    $arts[] = $plt->plot($steps, fitplot($hostLa, $history['alpha'], 100, 100))[0];
    $legend[] = 'Alpha';
    if (count($history['updateStep']) > 0) {
        $updateSteps = $hostLa->array($history['updateStep']);
        $arts[] = $plt->plot($updateSteps, fitplot($hostLa, $history['actorLoss'], 100, 100))[0];
        $arts[] = $plt->plot($updateSteps, fitplot($hostLa, $history['criticLoss'], 100, 100))[0];
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

if (!rlEnvBool('RL_SKIP_DEMO')) {
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

}