<?php
namespace Rindow\RL\Agents\Agent\PPO;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;

class Runner
{
    private RolloutBuffer $buffer;

    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private PPOAgent $agent,
        private int $rolloutSteps = 2048,
        private float $gamma = 0.99,
        private float $gaeLambda = 0.95,
        private ?float $solvedReward = null,
        private mixed $rewardFunction = null,
        private bool $bootstrapTruncated = true,
    ) {
        $this->buffer = new RolloutBuffer(
            $la,
            $rolloutSteps,
            $agent->observationDimension(),
            $agent->isContinuous() ? $env->actionSpace()->shape()[0] : 1,
            $agent->isContinuous(),
            $agent->usesActionMask() ? $env->actionSpace()->n() : 0,
        );
    }

    public function evaluate(int $episodes = 10) : float
    {
        return $this->evaluateDetailed($episodes)['rawReward'];
    }

    /** @return array{rawReward:float,transformedReward:float,steps:float} */
    public function evaluateDetailed(int $episodes = 10) : array
    {
        $rawTotal = 0.0;
        $transformedTotal = 0.0;
        $stepTotal = 0;
        for ($episode = 0; $episode < $episodes; $episode++) {
            [$obs] = $this->evalEnv->reset();
            $done = false;
            while (!$done) {
                $action = $this->agent->selectActionDeterministic($obs);
                if (!$this->agent->isContinuous()) {
                    $action = $this->la->array($action, dtype:NDArray::int32);
                }
                $action = $this->agent->clipAction($action);
                $currentObs = $obs;
                [$obs, $reward, $terminated, $truncated] = $this->evalEnv->step($action);
                $done = $terminated || $truncated;
                $rawTotal += $reward;
                $transformedTotal += $this->rewardFunction === null
                    ? $reward
                    : ($this->rewardFunction)(
                        $currentObs,
                        (int)$this->la->scalar($action),
                        $obs,
                        $reward,
                        $terminated,
                        $truncated,
                    );
                $stepTotal++;
            }
        }
        return [
            'rawReward' => $rawTotal / $episodes,
            'transformedReward' => $transformedTotal / $episodes,
            'steps' => $stepTotal / $episodes,
        ];
    }

    public function train(
        int $totalSteps,
        int $evalEvery = 10_000,
        int $evalEpisodes = 10,
        ?string $bestModelFile = null,
    ) : array
    {
        $progress = new ProgressBar();
        $history = [
            'step'=>[], 'trainShaped'=>[], 'trainSteps'=>[],
            'evalReward'=>[], 'evalShaped'=>[], 'evalSteps'=>[],
            'policyLoss'=>[], 'valueLoss'=>[], 'entropy'=>[],
        ];
        [$obs] = $this->env->reset();
        $progress->start('Steps', $totalSteps, 50);
        $lastMetrics = ['policy_loss'=>0.0, 'value_loss'=>0.0, 'entropy'=>0.0];
        $best = -INF;
        $bestTransformed = -INF;
        $episodeShaped = 0.0;
        $episodeSteps = 0;
        $windowShaped = 0.0;
        $windowSteps = 0;
        $windowEpisodes = 0;
        $lastEpisodeEnd = true;
        $sdeNoiseAge = 0;

        for ($step = 1; $step <= $totalSteps; $step++) {
            if ($this->agent->usesSDE() && $sdeNoiseAge === 0) {
                $this->agent->resetExplorationNoise();
            }
            $progress->update($step);
            [$state, $actionMask] = $this->agent->parseObservation($obs);
            [$action, $logProb, $value] = $this->agent->selectActionFromState(
                $state, $actionMask
            );
            if ($this->agent->usesSDE()) {
                $sdeNoiseAge++;
                if ($this->agent->sdeSampleFreq() > 0
                    && $sdeNoiseAge >= $this->agent->sdeSampleFreq()) {
                    $sdeNoiseAge = 0;
                }
            }
            $envAction = $this->agent->isContinuous()
                ? $this->agent->clipAction($action)
                : $this->la->array($action, dtype:NDArray::int32);
            [$nextObs, $reward, $terminated, $truncated] = $this->env->step(
                $envAction
            );
            $trainingReward = $this->rewardFunction === null
                ? $reward
                : ($this->rewardFunction)($obs, $action, $nextObs, $reward, $terminated, $truncated);
            $terminalForValue = $terminated || ($truncated && !$this->bootstrapTruncated);
            $this->buffer->add(
                $state, $action, $trainingReward, $terminalForValue, $terminated || $truncated,
                $value, $logProb, $actionMask
            );
            $episodeShaped += $trainingReward;
            $episodeSteps++;
            $obs = $nextObs;
            $lastEpisodeEnd = $terminated || $truncated;
            if ($terminated || $truncated) {
                $windowShaped += $episodeShaped;
                $windowSteps += $episodeSteps;
                $windowEpisodes++;
                $episodeShaped = 0.0;
                $episodeSteps = 0;
                [$obs] = $this->env->reset();
            }

            if ($this->buffer->full() || $step === $totalSteps) {
                // Only the final state needs a fresh bootstrap value.  For
                // all earlier transitions, GAE reuses the value saved for the
                // following observation, avoiding one NN inference per step.
                $lastValue = $lastEpisodeEnd ? 0.0 : $this->agent->value($obs);
                $lastMetrics = $this->agent->update(
                    $this->buffer->finish($this->gamma, $this->gaeLambda, $lastValue)
                );
                // A PPO rollout always starts with a fresh exploration matrix.
                $sdeNoiseAge = 0;
            }
            if ($step % $evalEvery === 0 || $step === $totalSteps) {
                $evaluation = $this->evaluateDetailed($evalEpisodes);
                $score = $evaluation['rawReward'];
                $transformedScore = $evaluation['transformedReward'];
                $trainShaped = $windowEpisodes > 0 ? $windowShaped / $windowEpisodes : 0.0;
                $trainSteps = $windowEpisodes > 0 ? $windowSteps / $windowEpisodes : 0.0;
                $improved = $score > $best
                    || ($score === $best && $transformedScore > $bestTransformed);
                if ($improved) {
                    $best = $score;
                    $bestTransformed = $transformedScore;
                }
                foreach (['step'=>$step, 'trainShaped'=>$trainShaped, 'trainSteps'=>$trainSteps,
                    'evalReward'=>$score,
                    'evalShaped'=>$evaluation['transformedReward'],
                    'evalSteps'=>$evaluation['steps'],
                    'policyLoss'=>$lastMetrics['policy_loss'], 'valueLoss'=>$lastMetrics['value_loss'],
                    'entropy'=>$lastMetrics['entropy']] as $key => $value) {
                    $history[$key][] = $value;
                }
                $progress->clearProgressBar();
                $transformedText = $this->rewardFunction === null
                    ? ''
                    : sprintf(' | EvalShaped=%6.1f', $evaluation['transformedReward']);
                printf(
                    "Step %7d | TrainShaped=%6.1f | TrainSteps=%5.1f | EvalReward=%6.1f%s | EvalSteps=%5.1f | PolicyLoss=%+.3e | ValueLoss=%+.3e | Entropy=%.3f\n",
                    $step, $trainShaped, $trainSteps, $score, $transformedText, $evaluation['steps'],
                    $lastMetrics['policy_loss'], $lastMetrics['value_loss'], $lastMetrics['entropy']
                );
                if ($improved && $bestModelFile !== null) {
                    $this->agent->saveWeightsToFile($bestModelFile);
                    echo "Best model saved: {$bestModelFile}\n";
                }
                if ($this->solvedReward !== null && $score >= $this->solvedReward) {
                    echo "Solved: mean evaluation reward >= {$this->solvedReward}\n";
                    break;
                }
                $windowShaped = 0.0;
                $windowSteps = 0;
                $windowEpisodes = 0;
            }
        }
        echo "\nTraining finished. Best evaluation reward: {$best}  time: {$progress->laptimeString()}\n";
        return $history;
    }
}
