<?php
namespace Rindow\RL\Agents\Agent\A2C;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;
use Rindow\RL\Agents\ReplayBuffer\RolloutBuffer;

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
        /** fn(Environment $env, mixed $rawObservation, bool $reset): NDArray|array */
        private mixed $observationFunction = null,
    ) {
        if ($observationFunction !== null && !is_callable($observationFunction)) {
            throw new \InvalidArgumentException('observationFunction must be callable.');
        }
        $this->buffer = new RolloutBuffer(
            $la, $rolloutSteps, $agent->observationShape(),
            $agent->actionDimension(), $agent->isContinuous(),
            $agent->usesActionMask() ? $agent->actionDimension() : 0
        );
    }

    private function networkObservation(Env $env, mixed $observation, bool $reset=false) : mixed
    {
        return $this->observationFunction === null
            ? $observation
            : ($this->observationFunction)($env,$observation,$reset);
    }

    public function evaluate(int $episodes = 10) : float
    {
        $total = 0.0;
        for ($episode = 0; $episode < $episodes; $episode++) {
            [$rawObservation] = $this->evalEnv->reset();
            $observation = $this->networkObservation(
                $this->evalEnv,$rawObservation,true
            );
            $done = false;
            while (!$done) {
                $action = $this->agent->selectActionDeterministic($observation);
                if (!$this->agent->isContinuous()) {
                    $action = $this->la->array($action, dtype:NDArray::int32);
                }
                [$rawObservation, $reward, $terminated, $truncated] = $this->evalEnv->step($action);
                $observation = $this->networkObservation(
                    $this->evalEnv,$rawObservation
                );
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
        [$rawObservation] = $this->env->reset();
        $observation = $this->networkObservation($this->env,$rawObservation,true);
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
            [$state, $actionMask] = $this->agent->parseObservation($observation);
            [$action, $value] = $this->agent->selectActionFromState($state, $actionMask);
            $envAction = $this->agent->isContinuous()
                ? $this->agent->clipAction($action)
                : $this->la->array($action, dtype:NDArray::int32);
            [$nextRawObservation, $reward, $terminated, $truncated] = $this->env->step(
                $envAction
            );
            $nextObservation = $this->networkObservation(
                $this->env,$nextRawObservation
            );
            $trainingReward = $this->rewardFunction === null
                ? $reward
                : ($this->rewardFunction)(
                    $rawObservation,$action,$nextRawObservation,$reward,$terminated,$truncated
                );
            $terminalForValue = $terminated || ($truncated && !$this->bootstrapTruncated);
            $this->buffer->add($state, $action, $trainingReward, $terminalForValue,
                $terminated || $truncated, $value, actionMask:$actionMask);
            $episodeReward += $reward;
            $episodeSteps++;
            $observation = $nextObservation;
            $rawObservation = $nextRawObservation;
            $lastEpisodeEnd = $terminated || $truncated;
            if ($lastEpisodeEnd) {
                $windowReward += $episodeReward;
                $windowSteps += $episodeSteps;
                $windowEpisodes++;
                $episodeReward = 0.0;
                $episodeSteps = 0;
                [$rawObservation] = $this->env->reset();
                $observation = $this->networkObservation(
                    $this->env,$rawObservation,true
                );
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
