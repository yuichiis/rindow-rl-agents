<?php
namespace Rindow\RL\Agents\Agent\SAC;

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
        $la = $this->la; 

        $this->actor         = new GSDEActor($nn, $obs_dim, $act_dim);
        $this->critic        = new Critic($nn, $obs_dim, $act_dim, HIDDEN_DIM);
        $this->critic_target = new Critic($nn, $obs_dim, $act_dim, HIDDEN_DIM);

        # ダミー入力で build してから weights をコピー
        $dummy_obs = $this->g->Variable($la->zeros($la->alloc([1, $obs_dim])));
        $dummy_act = $this->g->Variable($la->zeros($la->alloc([1, $act_dim])));
        
        $this->actor->forward_train($dummy_obs);
        $this->critic->forward($dummy_obs, $dummy_act);
        $this->critic_target->forward($dummy_obs, $dummy_act);
        
        $critic_vars = $this->critic->trainableVariables();
        $critic_vars = $this->critic->variables();


        soft_update($this->g, $this->critic, $this->critic_target, 1.0);   # 完全コピー

        $this->actor_opt  = $nn->optimizers->Adam(lr: LR_ACTOR);
        $this->critic_opt = $nn->optimizers->Adam(lr: LR_CRITIC);
        $this->alpha_opt  = $nn->optimizers->Adam(lr: LR_ALPHA);

        # 自動エントロピー調整
        # PyTorch: torch.tensor(log(ALPHA_INIT), requires_grad=True)
        # TF:      tf.Variable(..., trainable=True)
        $this->target_entropy = -(float)$act_dim;
        $this->log_alpha = $this->g->Variable(
            $this->la->array([log(ALPHA_INIT)]),
            trainable:true, name:"log_alpha"
        );
    }

    # @property
    public function alpha() : Variable
    {
        return $this->g->exp($this->log_alpha);
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
        [$obs, $actions, $rewards, $next_obs, $dones] = $buffer->sample(BATCH_SIZE);

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
        $gamma_dones_q_next = $g->mul(GAMMA, $g->mul($one_minus_dones, $q_next));
        $target_q = $g->stopGradient($g->add($rewards_v, $gamma_dones_q_next));

        # ── [B] Critic 更新 ──────────────────────
        # PyTorch: critic_loss.backward(); critic_opt.step()
        $critic = $this->critic;
        $critic_loss = $this->nn->with($tape = $g->GradientTape(), function()
        use ($g, $critic, $obs_v, $actions_v, $target_q)
        {
            [$q1, $q2] = $critic->forward($obs_v, $actions_v);
            $critic_loss = $g->add(
                $g->reduceMean($g->square($g->sub($q1, $target_q))),
                $g->reduceMean($g->square($g->sub($q2, $target_q)))
            );
            return $critic_loss;
        });

        $critic_vars = $critic->trainableVariables();
        $critic_grads = $tape->gradient($critic_loss, $critic_vars);
        $this->critic_opt->update($critic_vars, $critic_grads);

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
            $actor_loss = $g->reduceMean($g->sub($g->mul($g->stopGradient($agent->alpha()), $log_pi), $g->minimum($q1_pi, $q2_pi)));
            return [$actor_loss,$log_pi];
        });
        
        $actor_vars = $this->actor->trainableVariables();
        $actor_grads = $tape->gradient($actor_loss, $actor_vars);
        $this->actor_opt->update($actor_vars, $actor_grads);

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
        soft_update($this->g, $this->critic, $this->critic_target, TAU);

        return [
            "critic_loss" => $critic_loss->value()->toArray(),
            "actor_loss"  => $actor_loss->value()->toArray(),
            "alpha"       => $this->alpha()->value()->toArray(),
        ];
    }
}
