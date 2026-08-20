# Training Runners

Every algorithm package includes a `Runner`. A runner combines environment
interaction, storage, updates, periodic deterministic evaluation, solved-state
tracking, progress output, and optional best-model saving.

- [On-policy runners](on-policy-runners.md): A2C, PPO, REINFORCE
- [Off-policy runners](off-policy-runners.md): DQN, DDPG, SAC
- Direct-update runners: Q-learning and True Online Sarsa(λ)
- [Evaluation behavior](evaluation.md)

`isSolved()` reports whether the configured reward threshold was met for
`solvedEvaluations` consecutive evaluations.

## Direct-update runners

The Q-learning and Sarsa runners are episode-based and share this training API:

```php
$runner->train($totalEpisodes, $evalEvery=50,
               $evalEpisodes=10, $bestModelFile=null);
```

Both return history arrays containing episode, training reward, evaluation
reward, and mean absolute TD error. Q-learning bootstraps from the best next
action. Sarsa calls `startEpisode()`, carries its selected next action into the
following transition, and maintains Dutch traces within the episode. A time
limit ends interaction but still permits a final-state bootstrap; true
termination does not.
