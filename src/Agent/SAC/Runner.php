<?php
namespace Rindow\RL\Agents\Agent\SAC;

class Runner
{
    private object $mo;
    private object $la;
    private Builder $nn;
    private Env $env;
    private Env $evalEnv;
    private SACGSDEAgent $agent;
    private ReplayBuffer $buffer;
    public function __construct(
    )
    {
        $this->mo = new MatrixOperator();
        $this->la = $mo->laRawMode();
        $this->nn = new NeuralNetworks($mo);
        $this->env = new Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0($la);
        $this->evalEnv = new Rindow\RL\Gym\ClassicControl\ContinuousMountainCar\ContinuousMountainCarV0($la);

        $stateShape = $env->observationSpace()->shape();
        $obs_dim = $stateShape[0];

        $actionSpace = $env->actionSpace();
        $act_dim = $actionSpace->shape()[0];

        $act_limit = 1.0; 

        echo "Env: MountainCarContinuous-v0  obs_dim={$obs_dim}  act_dim={$act_dim}  act_limit={$act_limit}\n";
        echo "gSDE latent_dim=" . GSDE_LATENT_DIM . "  reset_freq=" . GSDE_RESET_FREQ . "\n";

        $this->agent  = new SACGSDEAgent($nn, $obs_dim, $act_dim, $act_limit);
        $this->buffer = new ReplayBuffer($nn, BUFFER_SIZE, $obs_dim, $act_dim);

    }

    # ─────────────────────────────────────────────
    # 評価ループ
    # ─────────────────────────────────────────────
    private function evaluate(
        Env $env,
        int $n_episodes = EVAL_EPISODES
    ) : float
    {
        $la = $nn->backend()->primaryLA();
        $total = 0.0;
        for ($i = 0; $i < $n_episodes; $i++) {
            [$obs, $info] = $env->reset();
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
    public function train(
        ?int $numIterations=null, ?int $numRolloutSteps=null, ?int $maxSteps=null, ?array $metrics=null,
        ?int $evalInterval=null, ?int $numEvalEpisodes=null, ?int $logInterval=null,
        ?int $targetScore=null, ?int $numAchievements=null,
        ?int $verbose=null
    ) : array
    {
        $numIterations ??= 1000;
        $evalInterval ??= 100;
        $numEvalEpisodes ??= 0;
        $logInterval ??= 100;
        $verbose ??= 0;
        $env = $this->env;
        $agent = $this->agent;
        $metrics ??= [];
        $numAchievements ??= 5;


        [$obs,$info] = $env->reset();
        $W_noise = $agent->sample_noise();

        $episode_reward = 0.0;
        $episode_step   = 0;
        $episode_count  = 0;
        $best_eval      = -INF;
        $eval_history   = [];
        $consecutive_high_scores = 0;

        for ($step = 1; $step <= $numIterations; $step++) {

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
                [$obs,$info] = $env->reset();
                $W_noise = $agent->sample_noise();
                $episode_reward = 0.0;
                $episode_step   = 0;
            }

            if ($step >= START_STEPS && $step % UPDATE_EVERY == 0) {
                $agent->update($buffer);
            }

            if ($step % EVAL_EVERY == 0) {
                $mean_reward = $this->evaluate($this->evalEnv);
                $eval_history[] = $mean_reward;
                if (count($eval_history) > 10) {
                    array_shift($eval_history);
                }
                $marker = ($mean_reward > $best_eval) ? " ← best" : "";
                $best_eval = max($best_eval, $mean_reward);

                if ($mean_reward >= 80.0) {
                    $consecutive_high_scores++;
                } else {
                    $consecutive_high_scores = 0;
                }

                $history_str = implode(", ", array_map(fn($v) => sprintf("%+.2f", $v), $eval_history));

                printf(
                    "Step %7d | EvalReward=%+8.2f | Best=%+8.2f | Alpha=%0.4f | Consecutive80+=%d | History=[%s]%s\n",
                    $step,
                    $mean_reward,
                    $best_eval,
                    $agent->alpha()->value()->toArray()[0],
                    $consecutive_high_scores,
                    $history_str,
                    $marker
                );

                if ($mean_reward >= 90.0) {
                    echo "🎉 Solved! (Single evaluation mean reward >= 90)\n";
                    break;
                }
                if ($consecutive_high_scores >= 3) {
                    echo "🎉 Solved! (Consecutive 3 evaluations mean reward >= 80)\n";
                    break;
                }
            }
        }

        echo "\nTraining finished. Best eval reward: {$best_eval}\n";

        echo "\n─────────────────────────────────────────────\n";
        echo "Testing trained model (5 episodes)\n";
        echo "─────────────────────────────────────────────\n";
        $test_episodes = 5;
        $test_rewards = [];
        for ($i = 1; $i <= $test_episodes; $i++) {
            [$obs, $info] = $env->reset();
            $W_noise = $agent->sample_noise();
            $done = false;
            $step = 0;
            $ep_reward = 0.0;
            while (!$done) {
                if ($step % GSDE_RESET_FREQ == 0) {
                    $W_noise = $agent->sample_noise();
                }
                $action = $agent->select_action($obs, $W_noise);
                [$next_obs, $reward, $terminated, $truncated, $info] = $env->step($action);
                $done = $terminated || $truncated;
                $obs = $next_obs;
                $ep_reward += $reward;
                $step += 1;
            }
            $test_rewards[] = $ep_reward;
            printf("Test Episode %d: Reward = %+8.2f (Steps: %d)\n", $i, $ep_reward, $step);
        }
        $avg_test_reward = array_sum($test_rewards) / count($test_rewards);
        printf("Average Test Reward: %+8.2f\n", $avg_test_reward);
    }
}
