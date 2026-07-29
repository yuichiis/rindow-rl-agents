<?php
namespace Rindow\RL\Agents\Agent\SAC;

#   PyTorch の Critic(q1, q2) に対応。
#   TF では Functional API で 2 つの独立したサブモデルを保持する。
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

