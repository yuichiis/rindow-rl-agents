<?php
namespace Rindow\RL\Agents;

use Rindow\NeuralNetworks\Gradient\Variable;
use Interop\Polite\Math\Matrix\NDArray;

interface Distribution
{
    /**
     *   Create the layers and parameters that represent the distribution.
     *
     *   Subclasses must define this, but the arguments and return type vary between
     *   concrete classes.
     * 
     * @return array{Module,Parameter} 
     */
    //public function probaDistributionNet(...$args) : Module | array;
    public function probaDistributionNet(
        int $latent_dim,
        ?float $log_std_init=null,
        ?int $latent_sde_dim=null,
    ) : array;

    /**
     *   Set parameters of the distribution.
     *
     *  @return SelfDistribution
     */
    //public function probaDistribution(...$args) : SelfDistribution;
    public function probaDistribution(
        Variable $mean_actions,  // from the action layer output
        Variable $log_std,       // from trainable variable in the policy network
        Variable $latent_sde,    // from the layer before the action 
    ) : Distribution;

    /**
     *   Returns the log likelihood
     *
     * @param Variable  x  the taken action
     * @return Variable The log likelihood of the distribution
     */
    public function log_prob(Variable $x) : Variable;

    /**
     *   Returns Shannon's entropy of the probability
     *
     * @return ?Variable  the entropy, or None if no analytical form is known
     */
    public function entropy() : ?Variable;

    /**
     * Returns a sample from the probability distribution
     * 
     * @return Variable the stochastic action
     */
    public function sample() : Variable;

    /**
     * Returns the most likely action (deterministic output)
     * from the probability distribution
     * 
     * @return Variable the stochastic action
     */
    public function mode() : Variable;

    /**
     * Return actions according to the probability distribution.
     * 
     * @param bool $deterministic
     * @return Variable actions
     */
    public function get_actions(?bool $deterministic=null) : Variable;

    /**
     * Returns samples from the probability distribution
     * given its parameters.
     * 
     * @return Variable actions
     */
    //public function actions_from_params(...$args) : Variable;
    public function actions_from_params(
        Variable $mean_actions,
        Variable $log_std,
        Variable $latent_sde,
        ?bool $deterministic=null,
    ) : Variable;

    /**
     * Returns samples and the associated log probabilities
     * from the probability distribution given its parameters.
     * 
     * @return array{Variable,Variable} actions and log prob
     */
    //public function log_prob_from_params(...$args) : array;
    public function log_prob_from_params(
        Variable $mean_actions,
        Variable $log_std,
        Variable $latent_sde,
    ) : array;
}
