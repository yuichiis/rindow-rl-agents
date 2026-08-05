<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;

/** Single-process synchronous A2C training loop. */
class Runner
{
    private RolloutBuffer $buffer;

    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private A2CAgent $agent,
        int $rolloutSteps = 5,
        private float $gamma = 0.99,
        private float $gaeLambda = 1.0,
        private ?float $solvedReward = null,
        private bool $bootstrapTruncated = true,
        private mixed $rewardFunction = null,
    ) {
        $this->buffer = new RolloutBuffer(
            $la, $rolloutSteps, $agent->observationDimension(),
            $agent->actionDimension(), $agent->isContinuous()
        );
    }

    public function evaluate(int $episodes = 10) : float
    {
        $total = 0.0;
        for ($episode = 0; $episode < $episodes; $episode++) {
            [$observation] = $this->evalEnv->reset();
            $done = false;
            while (!$done) {
                $action = $this->agent->selectActionDeterministic($observation);
                if (!$this->agent->isContinuous()) {
                    $action = $this->la->array($action, dtype:NDArray::int32);
                }
                [$observation, $reward, $terminated, $truncated] = $this->evalEnv->step($action);
                $total += $reward;
                $done = $terminated || $truncated;
            }
        }
        return $total / $episodes;
    }

    public function train(int $totalSteps, int $evalEvery = 5_000, int $evalEpisodes = 10,
        ?string $bestModelFile = null) : array
    {
        if ($totalSteps < 1 || $evalEvery < 1 || $evalEpisodes < 1) {
            throw new \InvalidArgumentException('Training and evaluation counts must be positive.');
        }
        $progress = new ProgressBar();
        $history = ['step'=>[], 'trainReward'=>[], 'trainSteps'=>[], 'evalReward'=>[],
            'policyLoss'=>[], 'valueLoss'=>[], 'entropy'=>[], 'std'=>[]];
        [$observation] = $this->env->reset();
        $progress->start('Steps', $totalSteps, 50);
        $lastMetrics = ['policy_loss'=>0.0, 'value_loss'=>0.0, 'entropy'=>0.0, 'std'=>0.0];
        $best = -INF;
        $episodeReward = 0.0;
        $episodeSteps = 0;
        $windowReward = 0.0;
        $windowSteps = 0;
        $windowEpisodes = 0;
        $lastEpisodeEnd = true;

        for ($step = 1; $step <= $totalSteps; $step++) {
            $progress->update($step);
            [$action, $value] = $this->agent->selectAction($observation);
            $envAction = $this->agent->isContinuous()
                ? $this->agent->clipAction($action)
                : $this->la->array($action, dtype:NDArray::int32);
            [$nextObservation, $reward, $terminated, $truncated] = $this->env->step(
                $envAction
            );
            $trainingReward = $this->rewardFunction === null
                ? $reward
                : ($this->rewardFunction)(
                    $observation, $action, $nextObservation, $reward, $terminated, $truncated
                );
            $terminalForValue = $terminated || ($truncated && !$this->bootstrapTruncated);
            $this->buffer->add($observation, $action, $trainingReward, $terminalForValue,
                $terminated || $truncated, $value);
            $episodeReward += $reward;
            $episodeSteps++;
            $observation = $nextObservation;
            $lastEpisodeEnd = $terminated || $truncated;
            if ($lastEpisodeEnd) {
                $windowReward += $episodeReward;
                $windowSteps += $episodeSteps;
                $windowEpisodes++;
                $episodeReward = 0.0;
                $episodeSteps = 0;
                [$observation] = $this->env->reset();
            }

            if ($this->buffer->full() || $step === $totalSteps) {
                $lastValue = $lastEpisodeEnd ? 0.0 : $this->agent->value($observation);
                $lastMetrics = $this->agent->update(
                    $this->buffer->finish($this->gamma, $this->gaeLambda, $lastValue)
                );
            }
            if ($step % $evalEvery === 0 || $step === $totalSteps) {
                $score = $this->evaluate($evalEpisodes);
                $trainReward = $windowEpisodes > 0 ? $windowReward / $windowEpisodes : 0.0;
                $trainSteps = $windowEpisodes > 0 ? $windowSteps / $windowEpisodes : 0.0;
                $improved = $score > $best;
                if ($improved) $best = $score;
                foreach (['step'=>$step, 'trainReward'=>$trainReward, 'trainSteps'=>$trainSteps,
                    'evalReward'=>$score, 'policyLoss'=>$lastMetrics['policy_loss'],
                    'valueLoss'=>$lastMetrics['value_loss'], 'entropy'=>$lastMetrics['entropy'],
                    'std'=>$lastMetrics['std']]
                    as $key => $value) $history[$key][] = $value;
                $progress->clearProgressBar();
                $stdText = $this->agent->isContinuous()
                    ? sprintf(' | Std=%.3f', $lastMetrics['std']) : '';
                printf("Step %7d | TrainReward=%6.1f | TrainSteps=%5.1f | EvalReward=%6.1f | PolicyLoss=%+.3e | ValueLoss=%+.3e | Entropy=%.3f%s\n",
                    $step, $trainReward, $trainSteps, $score, $lastMetrics['policy_loss'],
                    $lastMetrics['value_loss'], $lastMetrics['entropy'], $stdText);
                if ($improved && $bestModelFile !== null) {
                    $this->agent->saveWeightsToFile($bestModelFile);
                    echo "Best model saved: {$bestModelFile}\n";
                }
                if ($this->solvedReward !== null && $score >= $this->solvedReward) {
                    echo "Solved: mean evaluation reward >= {$this->solvedReward}\n";
                    break;
                }
                $windowReward = 0.0;
                $windowSteps = 0;
                $windowEpisodes = 0;
            }
        }
        echo "\nTraining finished. Best evaluation reward: {$best}  time: {$progress->laptimeString()}\n";
        return $history;
    }
}
