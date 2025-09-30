<?php
namespace Rindow\RL\Agents\Distribution;

use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Distribution\Distribution as NNDistribution;

/**
 *   Distribution class for using generalized State Dependent Exploration (gSDE).
 *   Paper: https://arxiv.org/abs/2005.05719
 *
 *   It is used to create the noise exploration matrix and
 *   compute the log probability of an action with that noise.
 *
 *   :param action_dim: Dimension of the action space.
 *   :param full_std: Whether to use (n_features x n_actions) parameters
 *       for the std instead of only (n_features,)
 *   :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
 *       a positive standard deviation (cf paper). It allows to keep variance
 *       above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
 *   :param squash_output: Whether to squash the output using a tanh function,
 *       this ensures bounds are satisfied.
 *   :param learn_features: Whether to learn features for gSDE or not.
 *       This will enable gradients to be backpropagated through the features
 *       ``latent_sde`` in the code.
 *   :param epsilon: small value to avoid NaN due to numerical imprecision.
 * 
 */
class StateDependentNoiseDistribution extends AbstractDistribution
{
    public ?TanhBijector $bijector;
    public ?int $latent_sde_dim = null;
    //public ?Variable $mean_actions = null;
    //public ?Variable $log_std = null;
    public int $action_dim;
    public bool $use_expln;
    public bool $full_std;
    public float $epsilon;
    public bool $learn_features;

    public NNDistribution $distribution;
    public NNDistribution $weights_dist;
    protected Variable $latent_sde;
    public Variable $exploration_mat;
    public Variable $exploration_matrices;

    public function __construct(
        Builder $nn,
        int $action_dim,
        ?bool $full_std=null,
        ?bool $use_expln=null,
        ?bool $squash_output=null,
        ?bool $learn_features=null,
        ?float $epsilon=null,
    )
    {
        echo "class StateDependentNoiseDistribution\n";

        $full_std ??= true;
        $use_expln ??= false;
        $squash_output ??= false;
        $learn_features ??= false;
        $epsilon ??= 1e-6;

        parent::__construct($nn);
        $this->action_dim = $action_dim;
        $this->use_expln = $use_expln;
        $this->full_std = $full_std;
        $this->epsilon = $epsilon;
        $this->learn_features = $learn_features;
        if($squash_output) {
            $this->bijector = new TanhBijector($epsilon);
        } else {
            $this->bijector = null;
        }
    }

    public function distribution() : NNDistribution
    {
        return $this->distribution;
    }

    /**
     *   Get the standard deviation from the learned parameter
     *   (log of it by default). This ensures that the std is positive.
     * 
     */
    public function get_std(Variable $log_std) : Variable
    {
        echo "func get_std()\n";
        $g = $this->g;
        if($this->use_expln) {
            // From gSDE paper, it allows to keep variance
            // above zero and prevent it from growing too fast
            //  below_threshold = exp(log_std) * (log_std <= 0)
            $below_threshold = $g->mul($g->exp($log_std), $g->lessEqual($log_std,0));
            # Avoid NaN: zeros values that are below zero
            // safe_log_std = log_std * (log_std > 0) + epsilon
            $safe_log_std = $g->add($g->mul($log_std, $g->greator($log_std, 0)), $this->epsilon);
            // above_threshold = (log1p(safe_log_std) + 1.0) * (log_std > 0)
            $above_threshold = $g->mul($g->add($g->log1p($safe_log_std), 1.0), $g->greator($log_std, 0));
            // std = below_threshold + above_threshold
            $std = $g->add($below_threshold, $above_threshold);
        } else {
            // Use normal exponential
            $std = $g->exp($log_std);
        }

        if($this->full_std){
            return $std;
        }
        if($this->latent_sde_dim === null){
            throw new LogicException("latent_sde_dim is null");
        }
        # Reduce the number of parameters:
        $std = $g->mul($g->ones([$this->latent_sde_dim, $this->action_dim]), $std);
        return $std;
    }

    /**
     *  Sample weights for the noise exploration matrix,
     *  using a centered Gaussian distribution.
     * 
     */
    public function sample_weights(
        Variable $log_std,
        ?int $batch_size=null,
        ) : void
    {
        echo "func sample_weights()\n";
        $g = $this->g;
        $batch_size ??= 1;
        //echo "log_std:(".implode(',',$log_std->shape()).")\n";
        $std = $this->get_std($log_std);
        //echo "std:(".implode(',',$std->shape()).")\n";
        $this->weights_dist = $this->nn->distributions->Normal($g->zerosLike($std), $std);
        echo "save weights_dist\n";
        # Reparametrization trick to pass gradients
        $this->exploration_mat = $this->weights_dist->sample();
        echo "save exploration_mat:(".implode(',',$this->exploration_mat->shape()).")\n";
        # Pre-compute matrices in case of parallel exploration
        $this->exploration_matrices = $this->weights_dist->sample(batchShape:[$batch_size]);
        echo "save exploration_matrices:(".implode(',',$this->exploration_matrices->shape()).")\n";
    }

    /**
     *   Create the layers and parameter that represent the distribution:
     *   one output will be the deterministic action, the other parameter will be the
     *   standard deviation of the distribution that control the weights of the noise matrix.
     *
     *   :param latent_dim: Dimension of the last layer of the policy (before the action layer)
     *   :param log_std_init: Initial value for the log standard deviation
     *   :param latent_sde_dim: Dimension of the last layer of the features extractor
     *       for gSDE. By default, it is shared with the policy network.
     * 
     * @return array{Module, Parameter}
     */
    public function probaDistributionNet(
        int $latent_dim,
        ?float $log_std_init=null,
        ?int $latent_sde_dim=null,
    ) : array
    {
        echo "func probaDistributionNet()\n";
        $nn = $this->nn;
        $g = $this->g;
        $log_std_init ??= -2.0;
        # Network for the deterministic action, it represents the mean of the distribution
        $mean_actions_net = $nn->layers->Dense($this->action_dim);
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        $this->latent_sde_dim = ($latent_sde_dim === null) ? $latent_dim : $latent_sde_dim;
        //echo "latent_sde_dim: ".$this->latent_sde_dim."\n";
        //echo "action_dim: ".$this->action_dim."\n";
        # Reduce the number of parameters if needed
        //echo "full_std: "; var_dump($this->full_std);
        if($this->full_std) {
            $log_std = $g->ones([$this->latent_sde_dim, $this->action_dim]);
        } else {
            $log_std = $g->ones([$this->latent_sde_dim, 1]);
        }
        //echo "log_std:(".implode(',',$log_std->shape()).")\n";
        # Transform it to a parameter so it can be optimized
        $log_std = $g->Variable($g->mul($log_std, $log_std_init), trainable:true);
        # Sample an exploration matrix
        $this->sample_weights($log_std);
        return [$mean_actions_net, $log_std];
    }

    /**
     *  Create the probability distribution given its parameters (mean, std)
     *  self: SelfStateDependentNoiseDistribution
     * 
     */
    public function probaDistribution(
        Variable $mean_actions,  // from the action layer output
        Variable $log_std,       // from trainable variable in the policy network
        Variable $latent_sde,    // from the layer before the action 
    ) : StateDependentNoiseDistribution
    {
        echo "func probaDistribution()\n";
        $g = $this->g;
        # Stop gradient if we don't want to influence the features
        $this->latent_sde = ($this->learn_features) ? $latent_sde : $g->stopGradient($latent_sde);
        echo "save latent_sde\n";
        $variance = $g->matmul($g->square($this->latent_sde), $g->square($this->get_std($log_std)));
        $this->distribution = $this->nn->distributions->Normal($mean_actions, $g->sqrt($g->add($variance, $this->epsilon)));
        echo "save distribution\n";
        return $this;
    }

    /**
     * 
     */
    public function log_prob(Variable $actions) : Variable
    {
        echo "func log_prob()\n";
        if($this->bijector !== null) {
            $gaussian_actions = $this->bijector->inverse($actions);
        } else {
            $gaussian_actions = $actions;
        }
        # log likelihood for a gaussian
        $log_prob = $this->distribution->log_prob($gaussian_actions);
        # Sum along action dim
        $log_prob = $this->sum_independent_dims($log_prob);

        if($this->bijector !==null) {
            # Squash correction (from original SAC implementation)
            $log_prob = $g->sub($log_prob,$g->reduceSum($this->bijector->log_prob_correction($gaussian_actions), axis:1));
        }
        return $log_prob;
    }

    public function entropy() : ?Variable
    {
        echo "func entropy()\n";
        if($this->bijector !==null) {
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return null;
        }
        return $this->sum_independent_dims($this->distribution->entropy());
    }

    public function sample() : Variable
    {
        echo "func sample()\n";
        $g = $this->g;
        $noise = $this->get_noise($this->latent_sde);
        $actions = $g->add($this->distribution->mean(), $noise);
        if($this->bijector!==null) {
            return $this->bijector->forward($actions);
        }
        return $actions;
    }

    public function mode() : Variable
    {
        echo "func mode()\n";
        $actions = $this->distribution->mean();
        if($this->bijector!==null) {
            return $this->bijector->forward($actions);
        }
        return $actions;
    }

    public function get_noise(Variable $latent_sde) : Variable
    {
        echo "func get_noise()\n";
        $g = $this->g;
        $latent_sde = ($this->learn_features) ? $latent_sde : $g->stopGradient($latent_sde);
        # Default case: only one exploration matrix
        if(count($latent_sde) == 1 || count($latent_sde) != count($this->exploration_matrices)) {
            return $g->matmul($latent_sde, $this->exploration_mat);
        }
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        $latent_sde = $g->expandDims($latent_sde,axis:1);
        # (batch_size, 1, n_actions)
        //echo "latent_sde:(".implode(',',$latent_sde->shape()).")\n";
        //echo "exploration_matrices:(".implode(',',$this->exploration_matrices->shape()).")\n";
        // batch gemm
        $noise = $g->matmul($latent_sde, $this->exploration_matrices);
        return $g->squeeze($noise,axis:1);
    }

    public function actions_from_params(
        Variable $mean_actions,
        Variable $log_std,
        Variable $latent_sde,
        ?bool $deterministic=null,
    ) : Variable
    {
        echo "func actions_from_params()\n";
        $deterministic ??= false;
        # Update the proba distribution
        $this->proba_distribution($mean_actions, $log_std, $latent_sde);
        return $this->get_actions(deterministic:$deterministic);
    }

    /**
     * @return array {Variable,Variable}
     */
    public function log_prob_from_params(
        Variable $mean_actions,
        Variable $log_std,
        Variable $latent_sde,
    ) : array
    {
        echo "func log_prob_from_params()\n";
        $actions = $this->actions_from_params($mean_actions, $log_std, $latent_sde);
        $log_prob = $this->log_prob($actions);
        return [$actions, $log_prob];
    }
}