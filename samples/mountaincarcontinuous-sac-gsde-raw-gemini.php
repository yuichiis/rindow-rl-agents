<?php
# """
# MountainCarContinuous-v0  SAC + gSDE  (TensorFlow / GradientTape 版)
# =====================================================================
# 
# PyTorch 版からの対応表:
#     torch.nn.Module          → tf.keras.Model
#     nn.Parameter             → tf.Variable(trainable=True)
#     optimizer.zero_grad()
#     loss.backward()          → with tf.GradientTape() as tape:
#     optimizer.step()         →     grads = tape.gradient(loss, vars)
#                              →     opt.apply_gradients(zip(grads, vars))
#     torch.no_grad()          → tape 外で計算 + tf.stop_gradient()
#     torch.randn_like(x)      → tf.random.normal(tf.shape(x))
#     torch.einsum             → tf.einsum
#     F.mse_loss               → tf.reduce_mean((a - b) ** 2)
#     torch.min(a, b)          → tf.minimum(a, b)
#     tensor.clamp(min=v)      → tf.maximum(tensor, v)
# 
# 依存: pip install gymnasium tensorflow numpy
# """

//import numpy as np
//import tensorflow as tf
//import gymnasium as gym
//import random

require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;


# ─────────────────────────────────────────────
# ハイパーパラメータ
# ─────────────────────────────────────────────
const ENV_ID          = "MountainCarContinuous-v0";
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
const EVAL_EVERY      = 5_000;
const EVAL_EPISODES   = 5;

//print(f"TensorFlow version: {tf.__version__}")
//print(f"GPUs: {tf.config.list_physical_devices('GPU')}")


# ─────────────────────────────────────────────
# リプレイバッファ  (numpy のまま、変更なし)
# ─────────────────────────────────────────────
class ReplayBuffer
{
    private Builder $nn;
    private object $la;
    private object $g;
    private int $capacity;
    private int $obs_dim;
    private int $act_dim;
    private int $ptr;
    private int $size;
    private NDArray $obs;
    private NDArray $rewards;
    private NDArray $next_obs;
    private NDArray $dones;
    private NDArray $actions;

    public function __construct(
        Builder $nn,
        int $capacity,
        int $obs_dim,
        int $act_dim
        )
    {
        $la = $nn->backend()->primaryLA();
        $this->nn = $nn;
        $this->la = $la;
        $this->g = $this->nn->gradient();
        $this->capacity = $capacity;
        $this->ptr = 0;
        $this->size = 0;
        $this->obs      = $la->zeros([capacity, obs_dim], dtype:NDArray::float32);
        $this->rewards  = $la->zeros([capacity, 1],       dtype:NDArray::float32);
        $this->next_obs = $la->zeros([capacity, obs_dim], dtype:NDArray::float32);
        $this->dones    = $la->zeros([capacity, 1],       dtype:NDArray::float32);
        $this->actions  = $la->zeros([capacity, act_dim], dtype:NDArray::float32);
    }

    public function add(
        NDArray $obs,
        NDArray $action,
        float $reward,
        NDArray $next_obs,
        bool $done
        ) : void
    {
        $this->obs[$this->ptr]      = $obs;
        $this->actions[$this->ptr]  = $action;
        $this->rewards[$this->ptr]  = $reward;
        $this->next_obs[$this->ptr] = $next_obs;
        $this->dones[$this->ptr]    = $done;
        $this->ptr  = ($this->ptr + 1) % $this->capacity;
        $this->size = min($this->size + 1, $this->capacity);
    }

    public function sample(int $batch_size) : array
    {
        $idx = $this->la->randint(0, $this->size, shape: [$batch_size]);
        return [
            $this->obs[$idx],
            $this->actions[$idx],
            $this->rewards[$idx],
            $this->next_obs[$idx],
            $this->dones[$idx],
        ];
    }
}

# ─────────────────────────────────────────────
# gSDE Actor
# ─────────────────────────────────────────────
#    PyTorch 版との対応:
#        nn.Sequential(Linear, ReLU, ...)  → tf.keras.Sequential([Dense(...), ...])
#        nn.Parameter(tensor)              → tf.Variable(..., trainable=True)
#        forward_inference(obs, W_noise)   → そのまま同名メソッド
#        forward_train(obs)                → そのまま同名メソッド
#        sample_noise()                    → そのまま同名メソッド
class GSDEActor extends AbstractModel
{
    private object $la;
    private object $g;
    private int $act_dim;
    private int $latents_dim;
    private AbstractModel $phi_net;
    private AbstractModel $mu_head;
    private Variable $log_std;
    
    public function __construct(
        Builder $nn,
        int $obs_dim, int $act_dim, int $latent_dim = GSDE_LATENT_DIM)
    {
        parent::__construct($nn);
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        
        $this->act_dim    = $act_dim;
        $this->latents_dim = $latent_dim;

        $this->phi_net = $nn->models->Sequential([
            $nn->layers->Dense(HIDDEN_DIM, activation:"relu", input_shape:[$obs_dim]),
            $nn->layers->Dense($latent_dim, activation:"relu"),
        ]);

        $this->mu_head = $nn->models->Sequential([
            $nn->layers->Dense($act_dim, input_shape:[$latent_dim]),
        ]);

        $this->log_std = $this->g->Variable(
            $this->la->fill($this->la->alloc([$act_dim, $latent_dim],dtype:NDArray::float32),-1.0),
            trainable:true, name:"log_std"
        );
    }

    private function phi_and_mu(Variable $obs) : array
    {
        $phi = $this->phi_net->forward($obs);
        $mu  = $this->mu_head->forward($phi);
        return [$phi, $mu];
    }

    private function std_W() : Variable
    {
        return $this->g->exp($this->log_std);
    }

    public function sample_noise() : Variable
    {
        $g = $this->nn->gradient();
        $std = $this->std_W();
        $eps = $g->randomNormal($g->shape($std));
        return $g->mul($eps, $std);
    }

    public function forward_inference(Variable $obs, Variable $W_noise) : Variable
    {
        [$phi, $mu] = $this->phi_and_mu($obs);
        $phi_T = $this->g->transpose($phi);
        $matmul = $this->g->matmul($W_noise, $phi_T);
        $noise = $this->g->transpose($matmul);
        return $this->g->tanh($this->g->add($mu, $noise));
    }

    public function forward_train(Variable $obs) : array
    {
        [$phi, $mu] = $this->phi_and_mu($obs);
        $std_W   = $this->std_W();

        $B     = $obs->shape()[0];
        
        $eps   = $this->g->randomNormal([$B, $this->act_dim, $this->latents_dim]);
        
        $std_W_expanded = $this->g->expandDims($std_W, 0); 
        $W     = $this->g->mul($eps, $std_W_expanded);
        
        $phi_reshaped = $this->g->reshape($phi, [$B, $this->latents_dim, 1]);
        $matmul = $this->g->matmul($W, $phi_reshaped);
        $noise = $this->g->squeeze($matmul, 2);

        $x_t = $this->g->add($mu, $noise);
        $y_t = $this->g->tanh($x_t);

        $std_W_sq = $this->g->square($std_W);
        $phi_sq = $this->g->square($phi);
        $phi_sq_T = $this->g->transpose($phi_sq);
        $matmul_sq = $this->g->matmul($std_W_sq, $phi_sq_T);
        $sqrt = $this->g->sqrt($matmul_sq);
        $sigma_z = $this->g->transpose($sqrt);
        $sigma_z = $this->g->maximum($this->g->constant(1e-6), $sigma_z);

        $log_sigma = $this->g->log($sigma_z);
        $diff = $this->g->sub($x_t, $mu);
        $diff_sq = $this->g->square($diff);
        $sigma_z_sq = $this->g->square($sigma_z);
        $two_sigma_z_sq = $this->g->mul(2.0, $sigma_z_sq);
        $term3 = $this->g->div($diff_sq, $two_sigma_z_sq);
        
        $log_prob = $this->g->sub(-0.91893853320467, $log_sigma);
        $log_prob = $this->g->sub($log_prob, $term3);

        $y_t_sq = $this->g->square($y_t);
        $tanh_corr_inner = $this->g->add($this->g->sub(1.0, $y_t_sq), 1e-6);
        $tanh_corr = $this->g->log($tanh_corr_inner);
        $log_prob = $this->g->sub($log_prob, $tanh_corr);
        
        $log_prob = $this->g->reduceSum($log_prob, axis: -1, keepdims: true);

        return [$y_t, $log_prob];
    }

    public function call(Variable $obs) : array
    {
        return $this->forward_train($obs);
    }
}

# ─────────────────────────────────────────────
# Critic (Double Q)
# ─────────────────────────────────────────────
class QNetwork extends AbstractModel
{
    private object $g;
    private AbstractModel $model;

    public function __construct(Builder $nn, int $obs_dim, int $act_dim, int $hidden_dim)
    {
        parent::__construct($nn);
        $this->g = $nn->gradient();
        $this->model = $nn->models->Sequential([
            $nn->layers->Dense($hidden_dim, activation: 'relu', input_shape: [$obs_dim + $act_dim]),
            $nn->layers->Dense($hidden_dim, activation: 'relu'),
            $nn->layers->Dense(1),
        ]);
    }

    public function call(Variable $obs, Variable $action, ?bool $training=null) : Variable
    {
        $x = $this->g->concat([$obs, $action], axis: -1);
        return $this->model->forward($x, $training);
    }
}

class Critic extends AbstractModel
{
    public QNetwork $q1;
    public QNetwork $q2;

    public function __construct(Builder $nn, int $obs_dim, int $act_dim, int $hidden_dim)
    {
        parent::__construct($nn);
        $this->q1 = new QNetwork($nn, $obs_dim, $act_dim, $hidden_dim);
        $this->q2 = new QNetwork($nn, $obs_dim, $act_dim, $hidden_dim);
    }

    public function call(Variable $obs, Variable $action, ?bool $training=null) : array
    {
        return [$this->q1->forward($obs, $action, $training), $this->q2->forward($obs, $action, $training)];
    }
}

# ─────────────────────────────────────────────
# ソフトアップデートユーティリティ
# ─────────────────────────────────────────────
function soft_update(object $g, AbstractModel $source, AbstractModel $target, float $tau) : void
{
    $src_vars = $source->trainableVariables();
    $tgt_vars = $target->trainableVariables();
    foreach($src_vars as $i => $src_w) {
        $tgt_w = $tgt_vars[$i];
        $scaled_src = $g->scale($tau, $src_w);
        $scaled_tgt = $g->scale(1.0 - $tau, $tgt_w);
        $new_val = $g->add($scaled_src, $scaled_tgt);
        $tgt_w->assign($new_val);
    }
}

# ─────────────────────────────────────────────
# SAC + gSDE エージェント
# ─────────────────────────────────────────────
class SACGSDEAgent
{
    private Builder $nn;
    private object $la;
    private object $g;
    private int $act_dim;
    private float $act_limit;
    public GSDEActor $actor;
    public Critic $critic;
    public Critic $critic_target;
    private float $target_entropy;
    private Variable $log_alpha;
    private object $actor_opt;
    private object $critic_opt;
    private object $alpha_opt;

    public function __construct(
        Builder $nn,
        int $obs_dim,
        int $act_dim,
        float $act_limit
    )
    {
        $this->nn = $nn;
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        $this->act_dim   = $act_dim;
        $this->act_limit = $act_limit;

        $this->actor         = new GSDEActor($nn, $obs_dim, $act_dim);
        $this->critic        = new Critic($nn, $obs_dim, $act_dim, HIDDEN_DIM);
        $this->critic_target = new Critic($nn, $obs_dim, $act_dim, HIDDEN_DIM);

        $dummy_obs = $this->g->Variable($this->la->zeros([1, $obs_dim]));
        $dummy_act = $this->g->Variable($this->la->zeros([1, $act_dim]));
        
        $this->actor->forward_train($dummy_obs);
        $this->critic->forward($dummy_obs, $dummy_act);
        $this->critic_target->forward($dummy_obs, $dummy_act);
        
        soft_update($this->g, $this->critic, $this->critic_target, 1.0);

        $this->actor_opt  = $nn->optimizers->Adam(lr: LR_ACTOR);
        $this->critic_opt = $nn->optimizers->Adam(lr: LR_CRITIC);
        $this->alpha_opt  = $nn->optimizers->Adam(lr: LR_ALPHA);

        $this->target_entropy = -(float)$act_dim;
        $this->log_alpha = $this->g->Variable(
            $this->la->array([log(ALPHA_INIT)]),
            trainable:true, name:"log_alpha"
        );
    }

    public function alpha() : Variable
    {
        return $this->g->exp($this->log_alpha);
    }

    public function sample_noise() : Variable
    {
        return $this->actor->sample_noise();
    }

    public function select_action(NDArray $obs, Variable $W_noise) : NDArray
    {
        $obs_t  = $this->g->Variable($this->la->expandDims($obs, 0));
        $action_var = $this->actor->forward_inference($obs_t, $W_noise);
        $action = $action_var->value();
        
        $action_flat = $this->la->reshape($action, [$this->act_dim]);
        $action_sc = $this->la->scal($this->act_limit, $action_flat);
        
        return $this->clip_ndarray($action_sc, -$this->act_limit, $this->act_limit);
    }
    
    private function clip_ndarray(NDArray $x, float $min, float $max) : NDArray
    {
        $arr = $x->toArray();
        array_walk_recursive($arr, function(&$v) use ($min, $max) {
            $v = max(min($v, $max), $min);
        });
        return $this->la->array($arr);
    }

    public function update(ReplayBuffer $buffer) : array
    {
        [$obs, $actions, $rewards, $next_obs, $dones] = $buffer->sample(BATCH_SIZE);

        $obs_v      = $this->g->Variable($obs);
        $actions_v  = $this->g->Variable($actions);
        $rewards_v  = $this->g->Variable($rewards);
        $next_obs_v = $this->g->Variable($next_obs);
        $dones_v    = $this->g->Variable($dones);

        [$next_actions, $next_log_pi] = $this->actor->forward_train($next_obs_v);
        $next_actions_sc = $this->g->mul($next_actions, $this->act_limit);
        
        [$q1_next, $q2_next] = $this->critic_target->forward($next_obs_v, $next_actions_sc);
        $q_next_min = $this->g->minimum($q1_next, $q2_next);
        
        $alpha_next_log_pi = $this->g->mul($this->alpha(), $next_log_pi);
        $q_next = $this->g->sub($q_next_min, $alpha_next_log_pi);
        
        $one_minus_dones = $this->g->sub(1.0, $dones_v);
        $gamma_dones_q_next = $this->g->mul(GAMMA, $this->g->mul($one_minus_dones, $q_next));
        $target_q = $this->g->stopGradient($this->g->add($rewards_v, $gamma_dones_q_next));

        $tape = $this->g->GradientTape();
        $critic_vars = $this->critic->trainableVariables();
        $tape->watch($critic_vars);
        
        [$q1, $q2] = $this->critic->forward($obs_v, $actions_v);
        $critic_loss = $this->g->add(
            $this->g->reduceMean($this->g->square($this->g->sub($q1, $target_q))),
            $this->g->reduceMean($this->g->square($this->g->sub($q2, $target_q)))
        );
        
        $critic_grads = $tape->gradient($critic_loss, $critic_vars);
        $this->critic_opt->update($critic_vars, $critic_grads);

        $tape = $this->g->GradientTape();
        $actor_vars = $this->actor->trainableVariables();
        $tape->watch($actor_vars);
        
        [$new_actions, $log_pi] = $this->actor->forward_train($obs_v);
        $new_actions_sc = $this->g->mul($new_actions, $this->act_limit);
        [$q1_pi, $q2_pi] = $this->critic->forward($obs_v, $new_actions_sc);
        $actor_loss = $this->g->reduceMean($this->g->sub($this->g->mul($this->g->stopGradient($this->alpha()), $log_pi), $this->g->minimum($q1_pi, $q2_pi)));
        
        $actor_grads = $tape->gradient($actor_loss, $actor_vars);
        $this->actor_opt->update($actor_vars, $actor_grads);

        $tape = $this->g->GradientTape();
        $alpha_vars = [$this->log_alpha];
        $tape->watch($alpha_vars);
        
        $alpha_loss = $this->g->scale(-1.0, $this->g->reduceMean($this->g->mul($this->log_alpha, $this->g->stopGradient($this->g->add($log_pi, $this->target_entropy)))));
        $alpha_grads = $tape->gradient($alpha_loss, $alpha_vars);
        $this->alpha_opt->update($alpha_vars, $alpha_grads);

        soft_update($this->g, $this->critic, $this->critic_target, TAU);

        return [
            "critic_loss" => $critic_loss->value()->toArray(),
            "actor_loss"  => $actor_loss->value()->toArray(),
            "alpha"       => $this->alpha()->value()->toArray(),
        ];
    }
}

# ─────────────────────────────────────────────
# 評価ループ
# ─────────────────────────────────────────────
function evaluate(
    Builder $nn,
    SACGSDEAgent $agent,
    int $n_episodes = EVAL_EPISODES
) : float
{
    $la = $nn->backend()->primaryLA();
    $env = new Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0($la);
    $total = 0.0;
    for ($i = 0; $i < $n_episodes; $i++) {
        $obs = $env->reset();
        $W_noise = $agent->sample_noise();
        $done = false;
        $step = 0;
        while (!$done) {
            if ($step % GSDE_RESET_FREQ == 0) {
                $W_noise = $agent->sample_noise();
            }
            $action = $agent->select_action($obs, $W_noise);
            [$next_obs, $reward, $terminated, $truncated, $info] = $env->step($action);
            $done = $terminated || $truncated;
            $obs = $next_obs;
            $total += $reward;
            $step  += 1;
        }
    }
    return $total / $n_episodes;
}

# ─────────────────────────────────────────────
# メインループ
# ─────────────────────────────────────────────
function main()
{
    $mo = new MatrixOperator();
    $la = $mo->laRawMode();
    $nn = new NeuralNetworks($mo);

    $env = new Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0($la);
    
    $stateShape = $env->observationSpace()->shape();
    $obs_dim = $stateShape[0];
    
    $actionSpace = $env->actionSpace();
    $act_dim = $actionSpace->shape()[0];
    
    $act_limit = 1.0; 

    echo "Env: MountainCarContinuous-v0  obs_dim={$obs_dim}  act_dim={$act_dim}  act_limit={$act_limit}\n";
    echo "gSDE latent_dim=" . GSDE_LATENT_DIM . "  reset_freq=" . GSDE_RESET_FREQ . "\n";

    $agent  = new SACGSDEAgent($nn, $obs_dim, $act_dim, $act_limit);
    $buffer = new ReplayBuffer($nn, BUFFER_SIZE, $obs_dim, $act_dim);

    $obs  = $env->reset();
    $W_noise = $agent->sample_noise();

    $episode_reward = 0.0;
    $episode_step   = 0;
    $episode_count  = 0;
    $best_eval      = -INF;

    for ($step = 1; $step <= TOTAL_STEPS; $step++) {

        if ($episode_step % GSDE_RESET_FREQ == 0) {
            $W_noise = $agent->sample_noise();
        }

        if ($step < START_STEPS) {
            $action = $actionSpace->sample();
        } else {
            $action = $agent->select_action($obs, $W_noise);
        }

        [$next_obs, $reward, $terminated, $truncated, $info] = $env->step($action);
        $done = $terminated || $truncated;
        $episode_reward += $reward;
        $episode_step   += 1;

        $buffer->add($obs, $action, $reward, $next_obs, $terminated);
        $obs = $next_obs;

        if ($done) {
            $episode_count += 1;
            $obs  = $env->reset();
            $W_noise = $agent->sample_noise();
            $episode_reward = 0.0;
            $episode_step   = 0;
        }

        if ($step >= START_STEPS && $step % UPDATE_EVERY == 0) {
            $agent->update($buffer);
        }

        if ($step % EVAL_EVERY == 0) {
            $mean_reward = evaluate($nn, $agent);
            $marker = ($mean_reward > $best_eval) ? " ← best" : "";
            $best_eval = max($best_eval, $mean_reward);
            printf(
                "Step %7d | EvalReward=%+8.2f | Alpha=%0.4f | Episodes=%d%s\n",
                $step,
                $mean_reward,
                $agent->alpha()->value()->toArray()[0],
                $episode_count,
                $marker
            );
            if ($mean_reward >= 90.0) {
                echo "🎉 Solved! (mean reward >= 90)\n";
                break;
            }
        }
    }

    echo "\nTraining finished. Best eval reward: {$best_eval}\n";
}

main();
    