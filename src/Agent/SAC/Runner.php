<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\NeuralNetworks\Builder\Builder;

class Runner
{
    private object $mo;
    private object $la;
    private Builder $nn;
    private Env $env;
    private Env $evalEnv;
    private SACGSDEAgent $agent;
    private int $act_dim;
    private int $act_limit;
    private ReplayBuffer $buffer;


    public function __construct(
        object $la,
        Builder $nn,
        Env $env,
        Env $evalEnv,
        SACGSDEAgent $agent,
        int $obs_dim,
        int $act_dim,
        int $act_limit,
        int $buffer_size,
    )
    {
        $this->la = $la;
        $this->env = $env;
        $this->evalEnv = $evalEnv;
        $this->agent = $agent;
        $this->act_dim = $act_dim;
        $this->act_limit = $act_limit;

        $this->buffer = new ReplayBuffer($la, $buffer_size, $obs_dim, $act_dim);
    }

    # ─────────────────────────────────────────────
    # 評価ループ
    # ─────────────────────────────────────────────
    public function evaluate(
        SACGSDEAgent $agent,
        int $n_episodes,
        int $gsde_reset_freq,
        bool $with_exploration_noise = false,
    ) : float
    {
        $la = $this->la;
        // 評価用の開始状態列は学習用の乱数列から独立させる。
        $env = $this->evalEnv;
        $total = 0.0;
        for ($i = 0; $i < $n_episodes; $i++) {
            [$obs, $info] = $env->reset();
            $W_noise = $with_exploration_noise ? $agent->sample_noise() : null;
            $done = false;
            $step = 0;
            while (!$done) {
                if ($with_exploration_noise && $step % $gsde_reset_freq == 0) {
                    $W_noise = $agent->sample_noise();
                }
                if ($with_exploration_noise) {
                    $action = $agent->select_action($obs, $W_noise);
                } else {
                    $action = $agent->select_action_deterministic($obs);
                }
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
    public function train(
        int $total_steps,
        int $start_steps,
        int $update_every,
        int $gsde_reset_freq,
        int $eval_every,
        int $eval_episodes,
    )
    {
        $la = $this->la;
        $env = $this->env;
        $agent = $this->agent;
        $buffer = $this->buffer;
        
        [$obs,$info] = $env->reset();
        $W_noise = $agent->sample_noise();

        $episode_reward = 0.0;
        $episode_step   = 0;
        $episode_count  = 0;
        $best_eval      = -INF;

        for ($step = 1; $step <= $total_steps; $step++) {

            if ($episode_step % $gsde_reset_freq == 0) {
                $W_noise = $agent->sample_noise();
            }

            if ($step < $start_steps) {
                $action = $la->randomUniform([$this->act_dim], -$this->act_limit, $this->act_limit);
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
                [$obs,$info] = $env->reset();
                $W_noise = $agent->sample_noise();
                $episode_reward = 0.0;
                $episode_step   = 0;
            }

            if ($step >= $start_steps && $step % $update_every == 0) {
                $agent->update($buffer);
            }

            if ($step % $eval_every == 0) {
                $deterministic_reward = $this->evaluate($agent, $eval_episodes, $gsde_reset_freq, with_exploration_noise: false);
                $noisy_reward = $this->evaluate($agent, $eval_episodes, $gsde_reset_freq, with_exploration_noise: true);
                $diag = $agent->diagnostics();
                $marker = ($deterministic_reward > $best_eval) ? " ← best" : "";
                $best_eval = max($best_eval, $deterministic_reward);
                printf(
                    "Step %7d | EvalDet=%+8.2f | EvalgSDE=%+8.2f | Alpha=%0.4f | Episodes=%d%s\n",
                    $step,
                    $deterministic_reward,
                    $noisy_reward,
                    $agent->alpha()->value()->toArray()[0],
                    $episode_count,
                    $marker
                );
                printf(
                    "  Diag: mu=[%+.4f,%+.4f,%+.4f] log_std=[%+.4f,%+.4f,%+.4f] gradRMS(actor/critic)=[%.3e/%.3e] Q(data/pi/target)=[%+.4f/%+.4f/%+.4f]\n",
                    $diag['mu_mean'], $diag['mu_min'], $diag['mu_max'],
                    $diag['log_std_mean'], $diag['log_std_min'], $diag['log_std_max'],
                    $diag['actor_grad_rms'], $diag['critic_grad_rms'],
                    $diag['q_data_mean'], $diag['q_pi_mean'], $diag['target_q_mean']
                );
                printf("  Actor grad RMS by variable: %s\n", json_encode($diag['actor_grad_rms_by_var']));
                if ($deterministic_reward >= 90.0) {
                    echo "🎉 Solved! (deterministic mean reward >= 90)\n";
                    break;
                }
            }
        }

        echo "\nTraining finished. Best eval reward: {$best_eval}\n";
    }
}
