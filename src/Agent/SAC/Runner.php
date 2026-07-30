<?php
namespace Rindow\RL\Agents\Agent\SAC;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\NeuralNetworks\Builder\Builder;

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


    public function __construct(
        object $la,
        Builder $nn,
        Env $env,
        Env $evalEnv,
        SACGSDEAgent $agent,
        int $obsDim,
        int $actDim,
        float $actLimit,
        int $bufferSize,
        ?float $solvedReward = null,
    )
    {
        $this->la = $la;
        $this->env = $env;
        $this->evalEnv = $evalEnv;
        $this->agent = $agent;
        $this->actDim = $actDim;
        $this->actLimit = $actLimit;

        $this->buffer = new ReplayBuffer($la, $bufferSize, $obsDim, $actDim);
        $this->solvedReward = $solvedReward;
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
        $la = $this->la;
        // 評価用の開始状態列は学習用の乱数列から独立させる。
        $env = $this->evalEnv;
        $total = 0.0;
        for ($i = 0; $i < $nEpisodes; $i++) {
            [$obs, $info] = $env->reset();
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
                [$nextObs, $reward, $terminated, $truncated, $info] = $env->step($action);
                $done = $terminated || $truncated;
                $obs = $nextObs;
                $total += $reward;
                $step  += 1;
            }
        }
        return $total / $nEpisodes;
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
    )
    {
        $la = $this->la;
        $env = $this->env;
        $agent = $this->agent;
        $buffer = $this->buffer;
        $evalgSDE ??= false;
        
        [$obs,$info] = $env->reset();
        $wNoise = $agent->sampleNoise();

        $episodeReward = 0.0;
        $episodeStep   = 0;
        $episodeCount  = 0;
        $bestEval      = -INF;

        for ($step = 1; $step <= $totalSteps; $step++) {

            if ($episodeStep % $gsdeResetFreq == 0) {
                $wNoise = $agent->sampleNoise();
            }

            if ($step < $startSteps) {
                $action = $la->randomUniform([$this->actDim], -$this->actLimit, $this->actLimit);
            } else {
                $action = $agent->selectAction($obs, $wNoise);
            }

            [$nextObs, $reward, $terminated, $truncated, $info] = $env->step($action);
            $done = $terminated || $truncated;
            $episodeReward += $reward;
            $episodeStep   += 1;

            $buffer->add($obs, $action, $reward, $nextObs, $terminated);
            $obs = $nextObs;

            if ($done) {
                $episodeCount += 1;
                [$obs,$info] = $env->reset();
                $wNoise = $agent->sampleNoise();
                $episodeReward = 0.0;
                $episodeStep   = 0;
            }

            if ($step >= $startSteps && $step % $updateEvery == 0) {
                $agent->update($buffer);
            }

            if ($step % $evalEvery == 0) {
                $deterministicReward = $this->evaluate($agent, $evalEpisodes, $gsdeResetFreq, withExplorationNoise: false);
                $strEvalgSDE = "";
                if($evalgSDE) {
                    $noisyReward = $this->evaluate($agent, $evalEpisodes, $gsdeResetFreq, withExplorationNoise: true);
                    $strEvalgSDE = sprintf("| EvalgSDE=%+8.2f ",$noisyReward);
                }
                $diag = $agent->diagnostics();
                $marker = ($deterministicReward > $bestEval) ? " ← best" : "";
                $bestEval = max($bestEval, $deterministicReward);
                printf(
                    "Step %7d | EvalDet=%+8.2f %s| Alpha=%0.4f | Episodes=%d%s\n",
                    $step,
                    $deterministicReward,
                    $strEvalgSDE,
                    $agent->alpha()->value()->toArray()[0],
                    $episodeCount,
                    $marker
                );
                // printf(
                //     "  Diag: mu=[%+.4f,%+.4f,%+.4f] log_std=[%+.4f,%+.4f,%+.4f] gradRMS(actor/critic)=[%.3e/%.3e] Q(data/pi/target)=[%+.4f/%+.4f/%+.4f]\n",
                //     $diag['muMean'], $diag['muMin'], $diag['muMax'],
                //     $diag['logStdMean'], $diag['logStdMin'], $diag['logStdMax'],
                //     $diag['actorGradRms'], $diag['criticGradRms'],
                //     $diag['qDataMean'], $diag['qPiMean'], $diag['targetQMean']
                // );
                // printf("  Actor grad RMS by variable: %s\n", json_encode($diag['actorGradRmsByVar']));
                if ($this->solvedReward !== null && $deterministicReward >= $this->solvedReward) {
                //     echo "🎉 Solved! (deterministic mean reward >= {$this->solvedReward})\n";
                    break;
                }
            }
        }

        //echo "\nTraining finished. Best eval reward: {$bestEval}\n";
    }
}
