<?php
namespace Rindow\RL\Agents\Agent\DQN;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Util\ProgressBar;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

class Runner
{
    private ReplayBuffer $buffer;

    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private DQNAgent $agent,
        int|array $obsDim,
        int $bufferSize,
        private ?float $solvedReward=null,
        private int $solvedEvaluations=1,
        private mixed $rewardFunction=null,
        /** fn(Environment $env, mixed $rawObservation, bool $reset): NDArray|array */
        private mixed $observationFunction=null,
    ) {
        if ($solvedEvaluations < 1) {
            throw new \InvalidArgumentException('solvedEvaluations must be positive.');
        }
        if ($observationFunction !== null && !is_callable($observationFunction)) {
            throw new \InvalidArgumentException('observationFunction must be callable.');
        }
        $this->buffer = new ReplayBuffer(
            $la,$bufferSize,$obsDim,
            actionMaskDimension:$agent->usesActionMask() ? $agent->actionDimension() : 0
        );
    }

    private function networkObservation(
        Env $env,
        mixed $observation,
        bool $reset=false,
    ) : mixed
    {
        return $this->observationFunction === null
            ? $observation
            : ($this->observationFunction)($env,$observation,$reset);
    }

    public function evaluate(int $episodes) : float
    {
        return $this->evaluateDetailed($episodes)['rawReward'];
    }

    /** @return array{rawReward:float,transformedReward:float,steps:float} */
    public function evaluateDetailed(int $episodes) : array
    {
        if ($episodes < 1) throw new \InvalidArgumentException('episodes must be positive.');
        $rawTotal = 0.0;
        $transformedTotal = 0.0;
        $stepTotal = 0;
        for ($episode=0; $episode<$episodes; $episode++) {
            [$rawObservation] = $this->evalEnv->reset();
            $observation = $this->networkObservation($this->evalEnv,$rawObservation,true);
            $done = false;
            while (!$done) {
                $actionValue = $this->agent->selectActionDeterministic($observation);
                $action = $this->la->array($actionValue,dtype:NDArray::int32);
                $currentRawObservation = $rawObservation;
                [$rawObservation,$reward,$terminated,$truncated] = $this->evalEnv->step($action);
                $observation = $this->networkObservation($this->evalEnv,$rawObservation);
                $done = $terminated || $truncated;
                $rawTotal += $reward;
                $transformedTotal += $this->transformReward(
                    $currentRawObservation,$actionValue,$rawObservation,$reward,$terminated,$truncated
                );
                $stepTotal++;
            }
        }
        return [
            'rawReward'=>$rawTotal/$episodes,
            'transformedReward'=>$transformedTotal/$episodes,
            'steps'=>$stepTotal/$episodes,
        ];
    }

    private function transformReward(
        mixed $observation,
        int $action,
        mixed $nextObservation,
        float $reward,
        bool $terminated,
        bool $truncated,
    ) : float {
        return $this->rewardFunction === null
            ? $reward
            : ($this->rewardFunction)(
                $observation,$action,$nextObservation,$reward,$terminated,$truncated
            );
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
        $history = ['step'=>[],'episodes'=>[],'trainShaped'=>[],'trainSteps'=>[],
            'evalReward'=>[],'evalShaped'=>[],'evalSteps'=>[],
            'updateStep'=>[],'loss'=>[],'qValue'=>[],'epsilon'=>[]];
        [$rawObservation] = $this->env->reset();
        $observation = $this->networkObservation($this->env,$rawObservation,true);
        $episodeCount = 0;
        $bestEval = -INF;
        $solvedCount = 0;
        $bestTransformed = -INF;
        $episodeShaped = 0.0;
        $episodeSteps = 0;
        $windowShaped = 0.0;
        $windowSteps = 0;
        $windowEpisodes = 0;
        $progress = new ProgressBar();
        $progress->start('Steps',$totalSteps,50);
        for ($step=1; $step<=$totalSteps; $step++) {
            $progress->update($step);
            $fraction = min(1.0,$step/$epsilonDecaySteps);
            $epsilon = $epsilonStart+($epsilonEnd-$epsilonStart)*$fraction;
            [$state,$actionMask] = $this->agent->parseObservation($observation);
            $actionValue = $this->agent->selectActionFromState($state,$epsilon,$actionMask);
            $action = $this->la->array($actionValue,dtype:NDArray::int32);
            [$nextRawObservation,$reward,$terminated,$truncated] = $this->env->step($action);
            $nextObservation = $this->networkObservation($this->env,$nextRawObservation);
            $trainingReward = $this->transformReward(
                $rawObservation,$actionValue,$nextRawObservation,$reward,$terminated,$truncated
            );
            [$nextState,$nextActionMask] = $this->agent->parseObservation($nextObservation);
            $done = $terminated || $truncated;
            // A time-limit truncation still has a valid bootstrap value.
            $this->buffer->add(
                $state,$actionValue,$trainingReward,$nextState,$terminated,$nextActionMask
            );
            $episodeShaped += $trainingReward;
            $episodeSteps++;
            $observation = $nextObservation;
            $rawObservation = $nextRawObservation;
            if ($done) {
                $episodeCount++;
                $windowShaped += $episodeShaped;
                $windowSteps += $episodeSteps;
                $windowEpisodes++;
                $episodeShaped = 0.0;
                $episodeSteps = 0;
                [$rawObservation] = $this->env->reset();
                $observation = $this->networkObservation($this->env,$rawObservation,true);
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
                $evaluation = $this->evaluateDetailed($evalEpisodes);
                $eval = $evaluation['rawReward'];
                $evalShaped = $evaluation['transformedReward'];
                $trainShaped = $windowEpisodes > 0 ? $windowShaped/$windowEpisodes : 0.0;
                $trainSteps = $windowEpisodes > 0 ? $windowSteps/$windowEpisodes : 0.0;
                $history['step'][] = $step;
                $history['episodes'][] = $episodeCount;
                $history['trainShaped'][] = $trainShaped;
                $history['trainSteps'][] = $trainSteps;
                $history['evalReward'][] = $eval;
                $history['evalShaped'][] = $evalShaped;
                $history['evalSteps'][] = $evaluation['steps'];
                $improved = $eval>$bestEval || ($eval===$bestEval && $evalShaped>$bestTransformed);
                $marker = $improved ? ' <- best' : '';
                if ($improved) {
                    $bestEval = $eval;
                    $bestTransformed = $evalShaped;
                    if ($bestModelFile!==null) $this->agent->saveWeightsToFile($bestModelFile);
                }
                $progress->clearProgressBar();
                $transformedText = $this->rewardFunction === null
                    ? '' : sprintf(' | EvalShaped=%+8.2f',$evalShaped);
                printf(
                    "Step %7d | TrainShaped=%+8.2f | TrainSteps=%5.1f | EvalDet=%+8.2f%s | EvalSteps=%5.1f | Epsilon=%.3f | Episodes=%d%s\n",
                    $step,$trainShaped,$trainSteps,$eval,$transformedText,$evaluation['steps'],
                    $epsilon,$episodeCount,$marker
                );
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
                $windowShaped = 0.0;
                $windowSteps = 0;
                $windowEpisodes = 0;
            }
        }
        echo "\nTraining finished. Best eval reward: {$bestEval} time: {$progress->laptimeString()}\n";
        return $history;
    }
}
