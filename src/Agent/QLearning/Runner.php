<?php
namespace Rindow\RL\Agents\Agent\QLearning;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;

class Runner
{
    private bool $solved = false;

    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private QLearningAgent $agent,
        private ?float $solvedReward = null,
        private int $solvedEvaluations = 1,
    ) {
        if ($solvedEvaluations < 1) {
            throw new \InvalidArgumentException('solvedEvaluations must be positive.');
        }
    }

    public function evaluate(int $episodes = 10) : float
    {
        if ($episodes < 1) throw new \InvalidArgumentException('episodes must be positive.');
        $total = 0.0;
        for ($episode=0; $episode<$episodes; $episode++) {
            [$observation] = $this->evalEnv->reset();
            $done = false;
            while (!$done) {
                $action = $this->la->array(
                    $this->agent->selectActionDeterministic($observation), dtype:NDArray::int32
                );
                [$observation,$reward,$terminated,$truncated] = $this->evalEnv->step($action);
                $total += $reward;
                $done = $terminated || $truncated;
            }
        }
        return $total/$episodes;
    }

    /** @return array<string,array<int,int|float>> */
    public function train(
        int $totalEpisodes,
        int $evalEvery = 50,
        int $evalEpisodes = 10,
        ?string $bestModelFile = null,
    ) : array {
        if ($totalEpisodes<1 || $evalEvery<1 || $evalEpisodes<1) {
            throw new \InvalidArgumentException('Training and evaluation counts must be positive.');
        }
        $this->solved = false;
        $history = ['episode'=>[],'trainReward'=>[],'evalReward'=>[],'tdError'=>[]];
        $progress = new ProgressBar();
        $progress->start('Episodes',$totalEpisodes,50);
        $best = -INF;
        $solvedCount = 0;
        $windowReward = 0.0; $windowTdError = 0.0; $windowSteps = 0; $windowEpisodes = 0;
        for ($episode=1; $episode<=$totalEpisodes; $episode++) {
            $progress->update($episode);
            [$observation] = $this->env->reset();
            $done = false;
            while (!$done) {
                $actionValue = $this->agent->selectAction($observation);
                [$nextObservation,$reward,$terminated,$truncated] = $this->env->step(
                    $this->la->array($actionValue,dtype:NDArray::int32)
                );
                $done = $terminated || $truncated;
                // A time-limit truncation has a valid final-state bootstrap value.
                $delta = $this->agent->update(
                    $observation,$actionValue,$reward,$nextObservation,$terminated
                );
                $windowReward += $reward; $windowTdError += abs($delta); $windowSteps++;
                $observation = $nextObservation;
            }
            $windowEpisodes++;
            if ($episode%$evalEvery===0 || $episode===$totalEpisodes) {
                $score = $this->evaluate($evalEpisodes);
                $trainReward = $windowReward/$windowEpisodes;
                $meanTdError = $windowTdError/$windowSteps;
                $improved = $score>$best;
                if ($improved) $best = $score;
                if ($this->solvedReward!==null) {
                    $solvedCount = $score >= $this->solvedReward ? $solvedCount+1 : 0;
                }
                $marker = $improved ? ' | Best' : '';
                $solvedText = $this->solvedReward === null
                    ? '' : " | SolvedCount={$solvedCount}/{$this->solvedEvaluations}";
                foreach (['episode'=>$episode,'trainReward'=>$trainReward,
                    'evalReward'=>$score,'tdError'=>$meanTdError] as $key=>$value) {
                    $history[$key][] = $value;
                }
                $progress->clearProgressBar();
                printf("Episode %5d | TrainReward=%7.1f | EvalReward=%7.1f | MeanAbsTD=%.4f%s%s\n",
                    $episode,$trainReward,$score,$meanTdError,$solvedText,$marker);
                if ($improved && $bestModelFile!==null) {
                    $this->agent->saveWeightsToFile($bestModelFile);
                    echo "Best model saved: {$bestModelFile}\n";
                }
                if ($this->solvedReward!==null
                    && $solvedCount >= $this->solvedEvaluations) {
                    echo "Solved: EvalReward >= {$this->solvedReward} for "
                        ."{$this->solvedEvaluations} consecutive evaluations\n";
                    $this->solved = true;
                    break;
                }
                $windowReward=0.0; $windowTdError=0.0; $windowSteps=0; $windowEpisodes=0;
            }
        }
        echo "\nTraining finished. BestEvalReward={$best} | Time={$progress->laptimeString()}\n";
        return $history;
    }

    public function isSolved() : bool
    {
        return $this->solved;
    }
}
