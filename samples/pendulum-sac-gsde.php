<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Pendulum\PendulumV1;

use Rindow\RL\Agents\Agent\SAC\SACGSDEAgent;
use Rindow\RL\Agents\Agent\SAC\Runner;

# ─────────────────────────────────────────────
# ハイパーパラメータ
# ─────────────────────────────────────────────
const ENV_ID          = "Pendulum-v1";
const SEED            = 42;
const TOTAL_STEPS     = 100000;
const START_STEPS     = 1000;
const BATCH_SIZE      = 256;
const BUFFER_SIZE     = 100000;
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

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$la->setSeed(SEED);
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

// 短縮診断用。未指定時は従来の定数を使用する。
$total_steps = (int)(getenv('RL_TOTAL_STEPS') ?: TOTAL_STEPS);
$eval_every = (int)(getenv('RL_EVAL_EVERY') ?: EVAL_EVERY);

$env = new PendulumV1($la);
$evalEnv = new PendulumV1($la);
$stateShape = $env->observationSpace()->shape();
$obs_dim = $stateShape[0];
$actionSpace = $env->actionSpace();
$act_dim = $actionSpace->shape()[0];
$env->observationSpace()->seed(SEED);
$env->actionSpace()->seed(SEED);
$act_limit = 1.0; 

echo "gSDE latent_dim=" . GSDE_LATENT_DIM . "  reset_freq=" . GSDE_RESET_FREQ . "\n";
echo "Env: ".ENV_ID."  obs_dim={$obs_dim}  act_dim={$act_dim}  act_limit={$act_limit}\n";

$agent  = new SACGSDEAgent(
    $nn,
    $obs_dim,
    $act_dim,
    $act_limit,
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
[$obs,$info] = $env->reset(seed: SEED);

$runner = new Runner(
    $la,
    $nn,
    $env,
    $evalEnv,
    $agent,
    $obs_dim,
    $act_dim,
    $act_limit,
    BUFFER_SIZE,
);


$runner->train(
    $total_steps,
    START_STEPS,
    UPDATE_EVERY,
    GSDE_RESET_FREQ,
    $eval_every,
    EVAL_EPISODES,
);


echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    [$obs, $info] = $env->reset();
    $env->render();
    $done=false;
    $step = 0;
    $total = 0.0;
    while (!$done) {
        $action = $agent->select_action_deterministic($obs);
        [$next_obs, $reward, $terminated, $truncated, $info] = $env->step($action);
        $done = $terminated || $truncated;
        $obs = $next_obs;
        $total += $reward;
        $step  += 1;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$i}, Steps: {$step}, Total Reward: {$total}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'\\pendulum-sac-gsde-trained.gif');
echo "filename: {$filename}\n";
