<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Util\ProgressBar;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

class Runner
{
    private ReplayBuffer $buffer;
    private OrnsteinUhlenbeckNoise $noise;
    private bool $solved = false;

    /** @param int|array<int,int> $obsDim */
    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private DDPGAgent $agent,
        int|array $obsDim,
        private int $actDim,
        private float $actLimit,
        int $bufferSize,
        private ?float $solvedReward=null,
        float $noiseSigma=0.2,
        float $noiseTheta=0.15,
        float $noiseDt=0.01,
        /** fn(mixed $obs, NDArray $action, mixed $nextObs, float $reward, bool $terminated, bool $truncated): float */
        private mixed $rewardFunction=null,
        /** fn(Environment $env, mixed $rawObservation, bool $reset): NDArray|array */
        private mixed $observationFunction=null,
        private int $solvedEvaluations=1,
    ) {
        if ($solvedEvaluations < 1) {
            throw new \InvalidArgumentException('solvedEvaluations must be positive.');
        }
        if ($rewardFunction !== null && !is_callable($rewardFunction)) {
            throw new \InvalidArgumentException('rewardFunction must be callable.');
        }
        if ($observationFunction !== null && !is_callable($observationFunction)) {
            throw new \InvalidArgumentException('observationFunction must be callable.');
        }
        $this->buffer = new ReplayBuffer($la,$bufferSize,$obsDim,$actDim);
        $this->noise = new OrnsteinUhlenbeckNoise($la,$actDim,$noiseSigma,$noiseTheta,$noiseDt);
    }

    private function networkObservation(Env $env, mixed $observation, bool $reset=false) : mixed
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
            [$rawObs] = $this->evalEnv->reset();
            $obs = $this->networkObservation($this->evalEnv,$rawObs,true);
            $done = false;
            while (!$done) {
                $action = $this->agent->selectActionDeterministic($obs);
                $currentRawObs = $rawObs;
                [$rawObs,$reward,$terminated,$truncated] = $this->evalEnv->step($action);
                $obs = $this->networkObservation($this->evalEnv,$rawObs);
                $done = $terminated || $truncated;
                $rawTotal += $reward;
                $transformedTotal += $this->transformReward(
                    $currentRawObs,$action,$rawObs,$reward,$terminated,$truncated
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
        mixed $action,
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

    /** @return array<string,array<int,int|float>> */
    public function train(
        int $totalSteps,
        int $startSteps,
        int $updateAfter,
        int $updateEvery,
        int $evalEvery,
        int $evalEpisodes,
        ?string $bestModelFile=null,
    ) : array {
        if ($updateEvery < 1 || $evalEvery < 1) throw new \InvalidArgumentException('Update/evaluation intervals must be positive.');
        $this->solved = false;
        $history = ['step'=>[],'episodes'=>[],'trainShaped'=>[],'trainSteps'=>[],
            'evalReward'=>[],'evalShaped'=>[],'evalSteps'=>[],'updateStep'=>[],
            'actorLoss'=>[],'criticLoss'=>[]];
        [$rawObs] = $this->env->reset();
        $obs = $this->networkObservation($this->env,$rawObs,true);
        $episodeCount = 0; $episodeStep = 0; $bestEval = -INF; $bestTransformed = -INF;
        $solvedCount = 0;
        $episodeShaped = 0.0; $windowShaped = 0.0; $windowSteps = 0; $windowEpisodes = 0;
        $progress = new ProgressBar(); $progress->start('Steps',$totalSteps,50);
        for ($step=1; $step<=$totalSteps; $step++) {
            $progress->update($step);
            if ($step <= $startSteps) {
                $action = $this->la->randomUniform([$this->actDim],-$this->actLimit,$this->actLimit);
            } else {
                // OU output is expressed in environment action units.
                $action = $this->agent->selectAction($obs,$this->noise->sample());
            }
            [$nextRawObs,$reward,$terminated,$truncated] = $this->env->step($action);
            $nextObs = $this->networkObservation($this->env,$nextRawObs);
            $done = $terminated || $truncated;
            $trainingReward = $this->transformReward(
                $rawObs,$action,$nextRawObs,$reward,$terminated,$truncated
            );
            // Time-limit truncation is bootstrapped; true termination is not.
            $this->buffer->add($obs,$action,$trainingReward,$nextObs,$terminated);
            $episodeShaped += $trainingReward;
            $obs = $nextObs; $rawObs = $nextRawObs; $episodeStep++;
            if ($done) {
                $windowShaped += $episodeShaped;
                $windowSteps += $episodeStep;
                $windowEpisodes++;
                $episodeCount++; [$rawObs] = $this->env->reset();
                $obs = $this->networkObservation($this->env,$rawObs,true);
                $episodeStep = 0; $episodeShaped = 0.0; $this->noise->reset();
            }
            if ($step >= max($updateAfter,1) && $this->buffer->size() > 0 && $step%$updateEvery===0) {
                // Keep the update-to-data ratio at one even when updates are grouped.
                for ($i=0; $i<$updateEvery; $i++) {
                    $metrics = $this->agent->update($this->buffer);
                    $history['updateStep'][] = $step;
                    $history['actorLoss'][] = $metrics['actor_loss'];
                    $history['criticLoss'][] = $metrics['critic_loss'];
                }
            }
            if ($step%$evalEvery===0) {
                $evaluation = $this->evaluateDetailed($evalEpisodes);
                $eval = $evaluation['rawReward'];
                $evalShaped = $evaluation['transformedReward'];
                $trainShaped = $windowEpisodes > 0 ? $windowShaped/$windowEpisodes : 0.0;
                $trainSteps = $windowEpisodes > 0 ? $windowSteps/$windowEpisodes : 0.0;
                $history['step'][]=$step; $history['episodes'][]=$episodeCount;
                $history['trainShaped'][]=$trainShaped; $history['trainSteps'][]=$trainSteps;
                $history['evalReward'][]=$eval; $history['evalShaped'][]=$evalShaped;
                $history['evalSteps'][]=$evaluation['steps'];
                $improved = $eval>$bestEval || ($eval===$bestEval && $evalShaped>$bestTransformed);
                $marker = $improved ? ' | Best' : '';
                if ($improved) {
                    $bestEval=$eval; $bestTransformed=$evalShaped;
                    if ($bestModelFile!==null) {
                        $this->agent->saveWeightsToFile($bestModelFile);
                    }
                }
                if ($this->solvedReward!==null) {
                    $solvedCount = $eval >= $this->solvedReward ? $solvedCount+1 : 0;
                }
                $solvedText = $this->solvedReward === null
                    ? '' : " | SolvedCount={$solvedCount}/{$this->solvedEvaluations}";
                $progress->clearProgressBar();
                $transformedText = $this->rewardFunction === null
                    ? '' : sprintf(' | EvalShaped=%+8.2f',$evalShaped);
                printf(
                    "Step %7d | TrainShaped=%+8.2f | TrainSteps=%5.1f | EvalReward=%+8.2f%s | EvalSteps=%5.1f | Episodes=%d%s%s\n",
                    $step,$trainShaped,$trainSteps,$eval,$transformedText,
                    $evaluation['steps'],$episodeCount,$solvedText,$marker
                );
                if ($improved && $bestModelFile!==null) {
                    echo "Best model saved: {$bestModelFile}\n";
                }
                if ($this->solvedReward!==null
                    && $solvedCount >= $this->solvedEvaluations) {
                    echo "Solved: EvalReward >= {$this->solvedReward} for "
                        ."{$this->solvedEvaluations} consecutive evaluations\n";
                    $this->solved = true;
                    break;
                }
                $windowShaped = 0.0; $windowSteps = 0; $windowEpisodes = 0;
            }
        }
        echo "\nTraining finished. BestEvalReward={$bestEval} | Time={$progress->laptimeString()}\n";
        return $history;
    }

    public function isSolved() : bool
    {
        return $this->solved;
    }
}
