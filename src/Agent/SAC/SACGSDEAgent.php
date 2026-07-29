<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Model\AbstractModel;


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
    private array $last_actor_grads = [];
    private array $last_critic_grads = [];
    private ?Variable $last_log_pi = null;
    private ?Variable $last_q_data = null;
    private ?Variable $last_q_pi = null;
    private ?Variable $last_target_q = null;
    private float $gamma;
    private float $tau;
    private int $batch_size;

    public function __construct(
        Builder $nn,
        int $obs_dim,
        int $act_dim,
        float $act_limit,
        int $gsde_latent_dim,
        int $hidden_dim,
        float $lr_actor,
        float $lr_critic,
        float $lr_alpha,
        float $alpha_init,
        float $gamma,
        float $tau,
        int $batch_size,
    )
    {
        $this->nn = $nn;
        $this->la = $nn->backend()->primaryLA();
        $this->g = $nn->gradient();
        $this->act_dim   = $act_dim;
        $this->act_limit = $act_limit;
        $this->gamma = $gamma;
        $this->tau = $tau;
        $this->batch_size = $batch_size;
        $la = $this->la; 

        $this->actor         = new GSDEActor($nn, $obs_dim, $act_dim, $gsde_latent_dim, $hidden_dim);
        $this->critic        = new Critic($nn, $obs_dim, $act_dim, $hidden_dim);
        $this->critic_target = new Critic($nn, $obs_dim, $act_dim, $hidden_dim);

        # ダミー入力で build してから weights をコピー
        $dummy_obs = $this->g->Variable($la->zeros($la->alloc([1, $obs_dim])));
        $dummy_act = $this->g->Variable($la->zeros($la->alloc([1, $act_dim])));
        
        $this->actor->forward_train($dummy_obs);
        $this->critic->forward($dummy_obs, $dummy_act);
        $this->critic_target->forward($dummy_obs, $dummy_act);
        
        $critic_vars = $this->critic->trainableVariables();
        $critic_vars = $this->critic->variables();


        $this->soft_update($this->g, $this->critic, $this->critic_target, 1.0);   # 完全コピー

        $this->actor_opt  = $nn->optimizers->Adam(lr: $lr_actor);
        $this->critic_opt = $nn->optimizers->Adam(lr: $lr_critic);
        $this->alpha_opt  = $nn->optimizers->Adam(lr: $lr_alpha);

        # 自動エントロピー調整
        # PyTorch: torch.tensor(log(ALPHA_INIT), requires_grad=True)
        # TF:      tf.Variable(..., trainable=True)
        $this->target_entropy = -(float)$act_dim;
        $this->log_alpha = $this->g->Variable(
            $this->la->array([log($alpha_init)]),
            trainable:true, name:"log_alpha"
        );
    }

    # ─────────────────────────────────────────────
    # ソフトアップデートユーティリティ
    # ─────────────────────────────────────────────
    #   PyTorch:
    #       for p, p_tgt in zip(src.parameters(), tgt.parameters()):
    #           p_tgt.data.copy_(tau * p + (1-tau) * p_tgt)
    #   TF:
    #       source.weights / target.weights をペアで assign
    public function soft_update(object $g, AbstractModel $source, AbstractModel $target, float $tau) : void
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

    # @property
    public function alpha() : Variable
    {
        return $this->g->exp($this->log_alpha);
    }

    private function rms(array $values) : float
    {
        $sum = 0.0;
        $count = 0;
        array_walk_recursive($values, function($v) use (&$sum, &$count) {
            $sum += (float)$v * (float)$v;
            $count++;
        });
        return $count ? sqrt($sum / $count) : 0.0;
    }

    private function gradient_rms_list(array $grads) : array
    {
        return array_map(fn($v) => $this->rms($v->toArray()), $grads);
    }

    private function range(array $values) : array
    {
        $flat = [];
        array_walk_recursive($values, function($v) use (&$flat) { $flat[] = (float)$v; });
        return [min($flat), max($flat), count($flat) ? array_sum($flat) / count($flat) : 0.0];
    }

    public function diagnostics() : array
    {
        $obs = $this->g->Variable($this->la->array(
            [[0.0, 0.0], [-0.5, 0.0], [0.0, 0.02], [0.4, 0.0]],
            dtype:NDArray::float32
        ));
        $mu = $this->actor->diagnostic_mu($obs)->value()->toArray();
        [$mu_min, $mu_max, $mu_mean] = $this->range($mu);
        [$ls_min, $ls_max, $ls_mean] = $this->range($this->actor->diagnostic_log_std()->value()->toArray());
        [$lp_min, $lp_max, $lp_mean] = $this->last_log_pi
            ? $this->range($this->last_log_pi->value()->toArray())
            : [0.0, 0.0, 0.0];
        $sigma_z = $this->actor->diagnostic_sigma_z();
        [$sz_min, $sz_max, $sz_mean] = $sigma_z
            ? $this->range($sigma_z->value()->toArray())
            : [0.0, 0.0, 0.0];
        $q_data_mean = $this->last_q_data ? $this->range($this->last_q_data->value()->toArray())[2] : 0.0;
        $q_pi_mean = $this->last_q_pi ? $this->range($this->last_q_pi->value()->toArray())[2] : 0.0;
        $target_q_mean = $this->last_target_q ? $this->range($this->last_target_q->value()->toArray())[2] : 0.0;
        return [
            'mu_mean' => $mu_mean, 'mu_min' => $mu_min, 'mu_max' => $mu_max,
            'log_std_mean' => $ls_mean, 'log_std_min' => $ls_min, 'log_std_max' => $ls_max,
            'log_pi_mean' => $lp_mean, 'log_pi_min' => $lp_min, 'log_pi_max' => $lp_max,
            'sigma_z_mean' => $sz_mean, 'sigma_z_min' => $sz_min, 'sigma_z_max' => $sz_max,
            'q_data_mean' => $q_data_mean, 'q_pi_mean' => $q_pi_mean, 'target_q_mean' => $target_q_mean,
            'actor_grad_rms' => $this->rms(array_map(fn($v)=>$v->toArray(), $this->last_actor_grads)),
            'actor_grad_rms_by_var' => $this->gradient_rms_list($this->last_actor_grads),
            'critic_grad_rms' => $this->rms(array_map(fn($v)=>$v->toArray(), $this->last_critic_grads)),
        ];
    }


    # ── 行動選択 ────────────────────────────────
    public function sample_noise() : Variable
    {
        return $this->actor->sample_noise();
    }


    public function select_action(NDArray $obs, Variable $W_noise) : NDArray
    {
        $obs_t  = $this->g->Variable($this->la->expandDims($obs, 0));
        $action_var = $this->actor->forward_inference($obs_t, $W_noise);
        $action = $action_var->value();
        
        $action_flat = $action->reshape([$this->act_dim]);
        $action_sc = $this->la->scal($this->act_limit, $action_flat);
        
        return $this->clip_ndarray($action_sc, -$this->act_limit, $this->act_limit);
    }

    public function select_action_deterministic(NDArray $obs) : NDArray
    {
        # 評価用: 探索ノイズなしで行動を選ぶ。
        $obs_t  = $this->g->Variable($this->la->expandDims($obs, 0));
        $action_var = $this->actor->forward_deterministic($obs_t);
        $action = $action_var->value();

        $action_flat = $action->reshape([$this->act_dim]);
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

    
    # ── 学習 ────────────────────────────────────
    #   各ブロックが独立した GradientTape を持つ。
    #
    #   PyTorch → TF 対応:
    #       optimizer.zero_grad()           (不要: TF は毎回新しい tape)
    #       loss.backward()              →  grads = tape.gradient(loss, vars)
    #       optimizer.step()             →  opt.apply_gradients(zip(grads, vars))
    #       with torch.no_grad():        →  tape 外 + tf.stop_gradient()
    #
    #   Rindow-NN への移植時も同じ構造で書ける。
    public function update(ReplayBuffer $buffer) : array
    {
        $g = $this->g;
        [$obs, $actions, $rewards, $next_obs, $dones] = $buffer->sample($this->batch_size);

        $obs_v      = $g->Variable($obs);
        $actions_v  = $g->Variable($actions);
        $rewards_v  = $g->Variable($rewards);
        $next_obs_v = $g->Variable($next_obs);
        $dones_v    = $g->Variable($dones);

        # ── [A] target_q (勾配不要) ──────────────
        # tape 外で計算 → 自動的に勾配追跡なし
        # tf.stop_gradient で念のため勾配を遮断
        [$next_actions, $next_log_pi] = $this->actor->forward_train($next_obs_v);
        $next_actions_sc = $g->mul($next_actions, $this->act_limit);
        
        [$q1_next, $q2_next] = $this->critic_target->forward($next_obs_v, $next_actions_sc);
        $q_next_min = $g->minimum($q1_next, $q2_next);
        
        $alpha_next_log_pi = $g->mul($this->alpha(), $next_log_pi);
        $q_next = $g->sub($q_next_min, $alpha_next_log_pi);
        
        $one_minus_dones = $g->sub(1.0, $dones_v);
        $gamma_dones_q_next = $g->mul($this->gamma, $g->mul($one_minus_dones, $q_next));
        $target_q = $g->stopGradient($g->add($rewards_v, $gamma_dones_q_next));
        $this->last_target_q = $target_q;

        # ── [B] Critic 更新 ──────────────────────
        # PyTorch: critic_loss.backward(); critic_opt.step()
        $critic = $this->critic;
        $agent = $this;
        $critic_loss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $critic, $obs_v, $actions_v, $target_q, $agent)
        {
            [$q1, $q2] = $critic->forward($obs_v, $actions_v);
            $agent->last_q_data = $g->minimum($q1, $q2);
            $critic_loss = $g->add(
                $g->reduceMean($g->square($g->sub($q1, $target_q))),
                $g->reduceMean($g->square($g->sub($q2, $target_q)))
            );
            return $critic_loss;
        });

        $critic_vars = $critic->trainableVariables();
        $critic_grads = $tape->gradient($critic_loss, $critic_vars);
        $this->last_critic_grads = $critic_grads;
        $this->critic_opt->update($critic_vars, $critic_grads);
        $this->critic->sync_weight_caches();

        # ── [C] Actor 更新 ───────────────────────
        $act_limit = $this->act_limit;
        $actor = $this->actor;
        $critic = $this->critic;
        $agent = $this;
        [$actor_loss,$log_pi] = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $agent, $actor, $obs_v, $act_limit, $critic)
        {
            [$new_actions, $log_pi] = $actor->forward_train($obs_v);
            $new_actions_sc = $g->mul($new_actions, $act_limit);
            [$q1_pi, $q2_pi] = $critic->forward($obs_v, $new_actions_sc);
            $agent->last_q_pi = $g->minimum($q1_pi, $q2_pi);
            $actor_loss = $g->reduceMean($g->sub($g->mul($g->stopGradient($agent->alpha()), $log_pi), $g->minimum($q1_pi, $q2_pi)));
            return [$actor_loss,$log_pi];
        });
        
        $actor_vars = $this->actor->trainableVariables();
        $actor_grads = $tape->gradient($actor_loss, $actor_vars);
        $this->last_log_pi = $log_pi;
        $this->last_actor_grads = $actor_grads;
        $this->actor_opt->update($actor_vars, $actor_grads);
        $this->actor->sync_weight_caches();
        if (getenv('RL_FREEZE_LOG_STD') === '1') {
            $this->actor->reset_log_std();
        }

        # ── [D] Alpha 更新 ───────────────────────
        $log_alpha = $this->log_alpha;
        $target_entropy = $this->target_entropy;
        $alpha_loss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $log_alpha, $log_pi, $target_entropy)
        {
            $alpha_loss = $g->scale(-1.0, $g->reduceMean($g->mul($log_alpha, $g->stopGradient($g->add($log_pi, $target_entropy)))));
            return $alpha_loss;
        });
        $alpha_vars = [$this->log_alpha];
        $alpha_grads = $tape->gradient($alpha_loss, $alpha_vars);
        $this->alpha_opt->update($alpha_vars, $alpha_grads);

        # ── [E] Critic ソフトアップデート ────────
        $this->soft_update($this->g, $this->critic, $this->critic_target, $this->tau);
        $this->critic_target->sync_weight_caches();

        return [
            "critic_loss" => $critic_loss->value()->toArray(),
            "actor_loss"  => $actor_loss->value()->toArray(),
            "alpha"       => $this->alpha()->value()->toArray(),
        ];
    }
}
