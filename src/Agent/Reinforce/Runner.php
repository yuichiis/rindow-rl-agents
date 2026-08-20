<?php
namespace Rindow\RL\Agents\Agent\Reinforce;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;

/** Episode-based training loop for REINFORCE. */
class Runner
{
    private bool $solved = false;

    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private ReinforceAgent $agent,
        private float $gamma = 0.99,
        private bool $normalizeReturns = true,
        private ?float $solvedReward = null,
        private int $solvedEvaluations = 1,
    ) {
        if ($gamma < 0.0 || $gamma > 1.0) {
            throw new \InvalidArgumentException('gamma must be between zero and one.');
        }
        if ($solvedEvaluations < 1) {
            throw new \InvalidArgumentException('solvedEvaluations must be positive.');
        }
    }

    public function evaluate(int $episodes = 10) : float
    {
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

    /** @return array<string,array<int,int|float>> */
    public function train(
        int $totalEpisodes,
        int $evalEvery = 50,
        int $evalEpisodes = 10,
        ?string $bestModelFile = null,
    ) : array {
        if ($totalEpisodes < 1 || $evalEvery < 1 || $evalEpisodes < 1) {
            throw new \InvalidArgumentException('Training and evaluation counts must be positive.');
        }
        $this->solved = false;
        $progress = new ProgressBar();
        $history = ['episode'=>[], 'trainReward'=>[], 'evalReward'=>[],
            'policyLoss'=>[], 'entropy'=>[]];
        $progress->start('Episodes', $totalEpisodes, 50);
        $best = -INF;
        $solvedCount = 0;
        $windowReward = 0.0;
        $windowEpisodes = 0;
        $lastMetrics = ['policy_loss'=>0.0, 'entropy'=>0.0];

        for ($episode = 1; $episode <= $totalEpisodes; $episode++) {
            $progress->update($episode);
            [$observation] = $this->env->reset();
            $observations = [];
            $actions = [];
            $rewards = [];
            $done = false;
            while (!$done) {
                $action = $this->agent->selectAction($observation);
                $observations[] = $observation;
                $actions[] = $action;
                [$observation, $reward, $terminated, $truncated] = $this->env->step(
                    $this->la->array($action, dtype:NDArray::int32)
                );
                $rewards[] = (float)$reward;
                $done = $terminated || $truncated;
            }
            $returns = $this->discountedReturns($rewards);
            $lastMetrics = $this->agent->update(
                $this->la->stack($observations),
                $this->la->array($actions, dtype:NDArray::int32),
                $this->la->array($returns, dtype:NDArray::float32),
            );
            $windowReward += array_sum($rewards);
            $windowEpisodes++;

            if ($episode % $evalEvery === 0 || $episode === $totalEpisodes) {
                $score = $this->evaluate($evalEpisodes);
                $trainReward = $windowReward / $windowEpisodes;
                $improved = $score > $best;
                if ($improved) $best = $score;
                if ($this->solvedReward !== null) {
                    $solvedCount = $score >= $this->solvedReward ? $solvedCount+1 : 0;
                }
                $marker = $improved ? ' | Best' : '';
                $solvedText = $this->solvedReward === null
                    ? '' : " | SolvedCount={$solvedCount}/{$this->solvedEvaluations}";
                foreach (['episode'=>$episode, 'trainReward'=>$trainReward, 'evalReward'=>$score,
                    'policyLoss'=>$lastMetrics['policy_loss'], 'entropy'=>$lastMetrics['entropy']]
                    as $key => $value) $history[$key][] = $value;
                $progress->clearProgressBar();
                printf("Episode %5d | TrainReward=%6.1f | EvalReward=%6.1f | PolicyLoss=%+.3e | Entropy=%.3f%s%s\n",
                    $episode, $trainReward, $score, $lastMetrics['policy_loss'],
                    $lastMetrics['entropy'], $solvedText, $marker);
                if ($improved && $bestModelFile !== null) {
                    $this->agent->saveWeightsToFile($bestModelFile);
                    echo "Best model saved: {$bestModelFile}\n";
                }
                if ($this->solvedReward !== null
                    && $solvedCount >= $this->solvedEvaluations) {
                    echo "Solved: EvalReward >= {$this->solvedReward} for "
                        ."{$this->solvedEvaluations} consecutive evaluations\n";
                    $this->solved = true;
                    break;
                }
                $windowReward = 0.0;
                $windowEpisodes = 0;
            }
        }
        echo "\nTraining finished. BestEvalReward={$best} | Time={$progress->laptimeString()}\n";
        return $history;
    }

    public function isSolved() : bool
    {
        return $this->solved;
    }

    /**
     * @param array<int,float> $rewards
     * @return array<int,float>
     */
    private function discountedReturns(array $rewards) : array
    {
        $returns = array_fill(0, count($rewards), 0.0);
        $running = 0.0;
        for ($i = count($rewards) - 1; $i >= 0; $i--) {
            $running = $rewards[$i] + $this->gamma * $running;
            $returns[$i] = $running;
        }
        if ($this->normalizeReturns && count($returns) > 1) {
            $mean = array_sum($returns) / count($returns);
            $sumSquares = 0.0;
            foreach ($returns as $value) $sumSquares += ($value - $mean) ** 2;
            $std = sqrt($sumSquares / count($returns));
            foreach ($returns as $i => $value) {
                $returns[$i] = ($value - $mean) / ($std + 1.0e-8);
            }
        }
        return $returns;
    }
}
