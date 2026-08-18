<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\RL\Agents\Util\ProgressBar;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

class Runner
{
    private object $mo;
    private object $la;
    private Builder $nn;
    private Env $env;
    private Env $evalEnv;
    private SACGSDEAgent $agent;
    private int $actDim;
    private float $actLimit;
    private ReplayBuffer $buffer;
    private ?float $solvedReward;
    private int $solvedEvaluations;
    private mixed $rewardFunction;
    private mixed $observationFunction;
    private ProgressBar $progressBar;
    private bool $solved = false;

    public function __construct(
        object $la,
        Builder $nn,
        Env $env,
        Env $evalEnv,
        SACGSDEAgent $agent,
        int|array $obsDim,
        int $actDim,
        float $actLimit,
        int $bufferSize,
        ?float $solvedReward = null,
        mixed $rewardFunction = null,
        /** fn(Environment $env, mixed $rawObservation, bool $reset): NDArray */
        mixed $observationFunction = null,
        int $solvedEvaluations = 1,
    )
    {
        $this->la = $la;
        $this->nn = $nn;
        $this->env = $env;
        $this->evalEnv = $evalEnv;
        $this->agent = $agent;
        $this->actDim = $actDim;
        $this->actLimit = $actLimit;

        $this->buffer = new ReplayBuffer($la, $bufferSize, $obsDim, $actDim);
        $this->solvedReward = $solvedReward;
        if ($solvedEvaluations < 1) {
            throw new \InvalidArgumentException('solvedEvaluations must be positive.');
        }
        $this->solvedEvaluations = $solvedEvaluations;
        if ($rewardFunction !== null && !is_callable($rewardFunction)) {
            throw new \InvalidArgumentException('rewardFunction must be callable.');
        }
        $this->rewardFunction = $rewardFunction;
        if ($observationFunction !== null && !is_callable($observationFunction)) {
            throw new \InvalidArgumentException('observationFunction must be callable.');
        }
        $this->observationFunction = $observationFunction;
    }

    private function networkObservation(Env $env, mixed $observation, bool $reset=false) : mixed
    {
        return $this->observationFunction === null
            ? $observation
            : ($this->observationFunction)($env,$observation,$reset);
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

    /**
     * 評価ループ
     */
    public function evaluate(
        SACGSDEAgent $agent,
        int $nEpisodes,
        int $gsdeResetFreq,
        bool $withExplorationNoise = false,
    ) : float
    {
        return $this->evaluateDetailed(
            $agent,$nEpisodes,$gsdeResetFreq,$withExplorationNoise
        )['rawReward'];
    }

    /** @return array{rawReward:float,transformedReward:float,steps:float} */
    public function evaluateDetailed(
        SACGSDEAgent $agent,
        int $nEpisodes,
        int $gsdeResetFreq,
        bool $withExplorationNoise = false,
    ) : array
    {
        $la = $this->la;
        // 評価用の開始状態列は学習用の乱数列から独立させる。
        $env = $this->evalEnv;
        $total = 0.0;
        $transformedTotal = 0.0;
        $stepTotal = 0;
        for ($i = 0; $i < $nEpisodes; $i++) {
            [$rawObs, $info] = $env->reset();
            $obs = $this->networkObservation($env,$rawObs,true);
            $wNoise = $withExplorationNoise ? $agent->sampleNoise() : null;
            $done = false;
            $step = 0;
            while (!$done) {
                if ($withExplorationNoise && $step % $gsdeResetFreq == 0) {
                    $wNoise = $agent->sampleNoise();
                }
                if ($withExplorationNoise) {
                    $action = $agent->selectAction($obs, $wNoise);
                } else {
                    $action = $agent->selectActionDeterministic($obs);
                }
                $currentRawObs = $rawObs;
                [$rawObs, $reward, $terminated, $truncated, $info] = $env->step($action);
                $nextObs = $this->networkObservation($env,$rawObs);
                $done = $terminated || $truncated;
                $transformedTotal += $this->transformReward(
                    $currentRawObs,$action,$rawObs,$reward,$terminated,$truncated
                );
                $obs = $nextObs;
                $total += $reward;
                $step  += 1;
                $stepTotal += 1;
            }
        }
        return [
            'rawReward'=>$total/$nEpisodes,
            'transformedReward'=>$transformedTotal/$nEpisodes,
            'steps'=>$stepTotal/$nEpisodes,
        ];
    }

    /**
     * メインループ
     */
    public function train(
        int $totalSteps,
        int $startSteps,
        int $updateEvery,
        int $gsdeResetFreq,
        int $evalEvery,
        int $evalEpisodes,
        ?bool $evalgSDE=null,
        ?string $bestModelFile=null,
    ) : array
    {
        $la = $this->la;
        $env = $this->env;
        $agent = $this->agent;
        $buffer = $this->buffer;
        $evalgSDE ??= false;

        $this->solved = false;
        $this->progressBar = new ProgressBar();
        
        [$rawObs,$info] = $env->reset();
        $obs = $this->networkObservation($env,$rawObs,true);
        $wNoise = $agent->sampleNoise();

        $episodeReward = 0.0;
        $episodeStep   = 0;
        $episodeCount  = 0;
        $bestEval      = -INF;
        $bestTransformed = -INF;
        $solvedCount = 0;
        $history = [
            'step' => [],
            'episodes' => [],
            'evalDet' => [],
            'evalgSDE' => [],
            'evalShaped' => [],
            'alpha' => [],
            'updateStep' => [],
            'actorLoss' => [],
            'criticLoss' => [],
        ];

        $this->progressBar->start("Steps",$totalSteps,50);
        for ($step = 1; $step <= $totalSteps; $step++) {

            $this->progressBar->update($step);
            if ($episodeStep % $gsdeResetFreq == 0) {
                $wNoise = $agent->sampleNoise();
            }

            if ($step < $startSteps) {
                $action = $la->randomUniform([$this->actDim], -$this->actLimit, $this->actLimit);
            } else {
                $action = $agent->selectAction($obs, $wNoise);
            }

            [$nextRawObs, $reward, $terminated, $truncated, $info] = $env->step($action);
            $nextObs = $this->networkObservation($env,$nextRawObs);
            $done = $terminated || $truncated;
            $episodeReward += $reward;
            $episodeStep   += 1;

            $trainingReward = $this->transformReward(
                $rawObs,$action,$nextRawObs,$reward,$terminated,$truncated
            );
            $buffer->add($obs, $action, $trainingReward, $nextObs, $terminated);
            $obs = $nextObs;
            $rawObs = $nextRawObs;

            if ($done) {
                $episodeCount += 1;
                [$rawObs,$info] = $env->reset();
                $obs = $this->networkObservation($env,$rawObs,true);
                $wNoise = $agent->sampleNoise();
                $episodeReward = 0.0;
                $episodeStep   = 0;
            }

            if ($step >= $startSteps && $step % $updateEvery == 0) {
                $updateMetrics = $agent->update($buffer);
                $history['updateStep'][] = $step;
                $history['actorLoss'][] = $updateMetrics['actor_loss'];
                $history['criticLoss'][] = $updateMetrics['critic_loss'];
            }

            if ($step % $evalEvery == 0) {
                $evaluation = $this->evaluateDetailed(
                    $agent,$evalEpisodes,$gsdeResetFreq,withExplorationNoise:false
                );
                $deterministicReward = $evaluation['rawReward'];
                $strEvalgSDE = "";
                if($evalgSDE) {
                    $noisyReward = $this->evaluate($agent, $evalEpisodes, $gsdeResetFreq, withExplorationNoise: true);
                    $strEvalgSDE = sprintf(" | EvalExploration=%+8.2f",$noisyReward);
                }
                $diag = $agent->diagnostics();
                $history['step'][] = $step;
                $history['episodes'][] = $episodeCount;
                $history['evalDet'][] = $deterministicReward;
                $history['evalShaped'][] = $evaluation['transformedReward'];
                $history['alpha'][] = $agent->alphaValue();
                if ($evalgSDE) {
                    $history['evalgSDE'][] = $noisyReward;
                }
                $improved = $deterministicReward > $bestEval
                    || ($deterministicReward === $bestEval
                        && $evaluation['transformedReward'] > $bestTransformed);
                $marker = $improved ? ' | Best' : '';
                if ($improved) {
                    $bestEval = $deterministicReward;
                    $bestTransformed = $evaluation['transformedReward'];
                    if ($bestModelFile !== null) {
                        $agent->saveWeightsToFile($bestModelFile);
                    }
                }
                if ($this->solvedReward !== null) {
                    $solvedCount = $deterministicReward >= $this->solvedReward
                        ? $solvedCount+1 : 0;
                }
                $solvedText = $this->solvedReward === null
                    ? '' : " | SolvedCount={$solvedCount}/{$this->solvedEvaluations}";
                $this->progressBar->clearProgressBar();
                $shapedText = $this->rewardFunction === null
                    ? '' : sprintf(' | EvalShaped=%+8.2f',$evaluation['transformedReward']);
                printf(
                    "Step %7d | EvalReward=%+8.2f%s%s | Alpha=%0.4f | Episodes=%d%s%s\n",
                    $step,
                    $deterministicReward,
                    $strEvalgSDE,
                    $shapedText,
                    $agent->alphaValue(),
                    $episodeCount,
                    $solvedText,
                    $marker
                );
                if ($improved && $bestModelFile !== null) {
                    echo "Best model saved: {$bestModelFile}\n";
                }
                // printf(
                //     "  Diag: mu=[%+.4f,%+.4f,%+.4f] log_std=[%+.4f,%+.4f,%+.4f] gradRMS(actor/critic)=[%.3e/%.3e] Q(data/pi/target)=[%+.4f/%+.4f/%+.4f]\n",
                //     $diag['muMean'], $diag['muMin'], $diag['muMax'],
                //     $diag['logStdMean'], $diag['logStdMin'], $diag['logStdMax'],
                //     $diag['actorGradRms'], $diag['criticGradRms'],
                //     $diag['qDataMean'], $diag['qPiMean'], $diag['targetQMean']
                // );
                // printf("  Actor grad RMS by variable: %s\n", json_encode($diag['actorGradRmsByVar']));
                if ($this->solvedReward !== null
                    && $solvedCount >= $this->solvedEvaluations) {
                    echo "Solved: EvalReward >= {$this->solvedReward} for "
                        ."{$this->solvedEvaluations} consecutive evaluations\n";
                    $this->solved = true;
                    break;
                }

            }
        }

        echo "\nTraining finished. BestEvalReward={$bestEval} | Time={$this->progressBar->laptimeString()}\n";
        return $history;
    }

    public function isSolved() : bool
    {
        return $this->solved;
    }
}
