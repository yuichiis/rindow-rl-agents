<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;

class Runner
{
    private ReplayBuffer $buffer;

    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private DQNAgent $agent,
        int $obsDim,
        int $bufferSize,
        private ?float $solvedReward=null,
        private int $solvedEvaluations=1,
    ) {
        if ($solvedEvaluations < 1) {
            throw new \InvalidArgumentException('solvedEvaluations must be positive.');
        }
        $this->buffer = new ReplayBuffer($la,$bufferSize,$obsDim);
    }

    public function evaluate(int $episodes) : float
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
                $done = $terminated || $truncated;
                $total += $reward;
            }
        }
        return $total/$episodes;
    }

    public function train(
        int $totalSteps,
        int $learningStarts,
        int $trainEvery,
        int $evalEvery,
        int $evalEpisodes,
        float $epsilonStart=1.0,
        float $epsilonEnd=0.05,
        int $epsilonDecaySteps=50_000,
        ?string $bestModelFile=null,
    ) : array {
        if ($totalSteps < 1 || $learningStarts < 0 || $trainEvery < 1
            || $evalEvery < 1 || $epsilonDecaySteps < 1) {
            throw new \InvalidArgumentException('Invalid DQN training parameters.');
        }
        $history = ['step'=>[],'episodes'=>[],'evalReward'=>[],
            'updateStep'=>[],'loss'=>[],'qValue'=>[],'epsilon'=>[]];
        [$observation] = $this->env->reset();
        $episodeCount = 0;
        $bestEval = -INF;
        $solvedCount = 0;
        $progress = new ProgressBar();
        $progress->start('Steps',$totalSteps,50);
        for ($step=1; $step<=$totalSteps; $step++) {
            $progress->update($step);
            $fraction = min(1.0,$step/$epsilonDecaySteps);
            $epsilon = $epsilonStart+($epsilonEnd-$epsilonStart)*$fraction;
            $actionValue = $this->agent->selectAction($observation,$epsilon);
            $action = $this->la->array($actionValue,dtype:NDArray::int32);
            [$nextObservation,$reward,$terminated,$truncated] = $this->env->step($action);
            $done = $terminated || $truncated;
            // A time-limit truncation still has a valid bootstrap value.
            $this->buffer->add($observation,$actionValue,$reward,$nextObservation,$terminated);
            $observation = $nextObservation;
            if ($done) {
                $episodeCount++;
                [$observation] = $this->env->reset();
            }
            if ($step >= $learningStarts && $this->buffer->size() > 0
                && $step%$trainEvery === 0) {
                $metrics = $this->agent->update($this->buffer);
                $history['updateStep'][] = $step;
                $history['loss'][] = $metrics['loss'];
                $history['qValue'][] = $metrics['q_value'];
                $history['epsilon'][] = $epsilon;
            }
            if ($step%$evalEvery === 0) {
                $eval = $this->evaluate($evalEpisodes);
                $history['step'][] = $step;
                $history['episodes'][] = $episodeCount;
                $history['evalReward'][] = $eval;
                $marker = $eval>$bestEval ? ' <- best' : '';
                if ($eval>$bestEval) {
                    $bestEval = $eval;
                    if ($bestModelFile!==null) $this->agent->saveWeightsToFile($bestModelFile);
                }
                $progress->clearProgressBar();
                printf("Step %7d | EvalDet=%+8.2f | Epsilon=%.3f | Episodes=%d%s\n",
                    $step,$eval,$epsilon,$episodeCount,$marker);
                if ($this->solvedReward!==null) {
                    $solvedCount = $eval >= $this->solvedReward ? $solvedCount+1 : 0;
                    if ($solvedCount > 0) {
                        echo "Solved evaluations: {$solvedCount}/{$this->solvedEvaluations}\n";
                    }
                    if ($solvedCount >= $this->solvedEvaluations) {
                        echo "Solved! (deterministic mean reward >= {$this->solvedReward} "
                            ."for {$this->solvedEvaluations} consecutive evaluations)\n";
                        break;
                    }
                }
            }
        }
        echo "\nTraining finished. Best eval reward: {$bestEval} time: {$progress->laptimeString()}\n";
        return $history;
    }
}
