<?php
namespace Rindow\RL\Agents\Agent\Sarsa;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;

class Runner
{
    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private TrueOnlineSarsaLambdaAgent $agent,
        private ?float $solvedReward = null,
    ) {}

    public function evaluate(int $episodes = 10) : float
    {
        if ($episodes < 1) throw new \InvalidArgumentException('episodes must be positive.');
        $total = 0.0;
        for ($episode = 0; $episode < $episodes; $episode++) {
            [$observation] = $this->evalEnv->reset();
            $done = false;
            while (!$done) {
                $action = $this->la->array(
                    $this->agent->selectActionDeterministic($observation), dtype:NDArray::int32
                );
                [$observation, $reward, $terminated, $truncated] = $this->evalEnv->step($action);
                $total += $reward;
                $done = $terminated || $truncated;
            }
        }
        return $total / $episodes;
    }

    public function train(
        int $totalEpisodes,
        int $evalEvery = 50,
        int $evalEpisodes = 10,
        ?string $bestModelFile = null,
    ) : array {
        if ($totalEpisodes < 1 || $evalEvery < 1 || $evalEpisodes < 1) {
            throw new \InvalidArgumentException('Training and evaluation counts must be positive.');
        }
        $history = ['episode'=>[], 'trainReward'=>[], 'evalReward'=>[], 'tdError'=>[]];
        $progress = new ProgressBar();
        $progress->start('Episodes', $totalEpisodes, 50);
        $best = -INF;
        $windowReward = 0.0;
        $windowTdError = 0.0;
        $windowSteps = 0;
        $windowEpisodes = 0;

        for ($episode = 1; $episode <= $totalEpisodes; $episode++) {
            $progress->update($episode);
            $this->agent->startEpisode();
            [$observation] = $this->env->reset();
            $action = $this->agent->selectAction($observation);
            $done = false;
            while (!$done) {
                [$nextObservation, $reward, $terminated, $truncated] = $this->env->step(
                    $this->la->array($action, dtype:NDArray::int32)
                );
                $done = $terminated || $truncated;
                // Time-limit truncation is not an MDP terminal: bootstrap once from its final state.
                $terminalForValue = $terminated;
                $nextAction = $terminalForValue ? null : $this->agent->selectAction($nextObservation);
                $delta = $this->agent->update(
                    $observation, $action, $reward, $nextObservation, $nextAction, $terminalForValue
                );
                $windowReward += $reward;
                $windowTdError += abs($delta);
                $windowSteps++;
                $observation = $nextObservation;
                if (!$done) $action = $nextAction;
            }
            $windowEpisodes++;

            if ($episode % $evalEvery === 0 || $episode === $totalEpisodes) {
                $score = $this->evaluate($evalEpisodes);
                $trainReward = $windowReward / $windowEpisodes;
                $meanTdError = $windowSteps > 0 ? $windowTdError / $windowSteps : 0.0;
                $improved = $score > $best;
                if ($improved) $best = $score;
                foreach (['episode'=>$episode, 'trainReward'=>$trainReward,
                    'evalReward'=>$score, 'tdError'=>$meanTdError] as $key=>$value) {
                    $history[$key][] = $value;
                }
                $progress->clearProgressBar();
                printf("Episode %5d | TrainReward=%7.1f | EvalReward=%7.1f | MeanAbsTD=%.4f\n",
                    $episode, $trainReward, $score, $meanTdError);
                if ($improved && $bestModelFile !== null) {
                    $this->agent->saveWeightsToFile($bestModelFile);
                    echo "Best model saved: {$bestModelFile}\n";
                }
                if ($this->solvedReward !== null && $score >= $this->solvedReward) {
                    echo "Solved: mean evaluation reward >= {$this->solvedReward}\n";
                    break;
                }
                $windowReward = 0.0;
                $windowTdError = 0.0;
                $windowSteps = 0;
                $windowEpisodes = 0;
            }
        }
        echo "\nTraining finished. Best evaluation reward: {$best}  time: {$progress->laptimeString()}\n";
        return $history;
    }
}
