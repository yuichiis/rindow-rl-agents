<?php
namespace Rindow\RL\Agents\Agent\DDPG;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Util\ProgressBar;

class Runner
{
    private ReplayBuffer $buffer;
    private OrnsteinUhlenbeckNoise $noise;

    public function __construct(
        private object $la,
        private Env $env,
        private Env $evalEnv,
        private DDPGAgent $agent,
        int $obsDim,
        private int $actDim,
        private float $actLimit,
        int $bufferSize,
        private ?float $solvedReward=null,
        float $noiseSigma=0.2,
        float $noiseTheta=0.15,
        float $noiseDt=0.01,
    ) {
        $this->buffer = new ReplayBuffer($la,$bufferSize,$obsDim,$actDim);
        $this->noise = new OrnsteinUhlenbeckNoise($la,$actDim,$noiseSigma,$noiseTheta,$noiseDt);
    }

    public function evaluate(int $episodes) : float
    {
        $total = 0.0;
        for ($episode=0; $episode<$episodes; $episode++) {
            [$obs] = $this->evalEnv->reset(); $done = false;
            while (!$done) {
                $action = $this->agent->selectActionDeterministic($obs);
                [$obs,$reward,$terminated,$truncated] = $this->evalEnv->step($action);
                $done = $terminated || $truncated; $total += $reward;
            }
        }
        return $total/$episodes;
    }

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
        $history = ['step'=>[],'episodes'=>[],'evalReward'=>[],'updateStep'=>[],
            'actorLoss'=>[],'criticLoss'=>[]];
        [$obs] = $this->env->reset();
        $episodeCount = 0; $episodeStep = 0; $bestEval = -INF;
        $progress = new ProgressBar(); $progress->start('Steps',$totalSteps,50);
        for ($step=1; $step<=$totalSteps; $step++) {
            $progress->update($step);
            if ($step <= $startSteps) {
                $action = $this->la->randomUniform([$this->actDim],-$this->actLimit,$this->actLimit);
            } else {
                // OU output is expressed in environment action units.
                $action = $this->agent->selectAction($obs,$this->noise->sample());
            }
            [$nextObs,$reward,$terminated,$truncated] = $this->env->step($action);
            $done = $terminated || $truncated;
            // Time-limit truncation is bootstrapped; true termination is not.
            $this->buffer->add($obs,$action,$reward,$nextObs,$terminated);
            $obs = $nextObs; $episodeStep++;
            if ($done) {
                $episodeCount++; [$obs] = $this->env->reset();
                $episodeStep = 0; $this->noise->reset();
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
                $eval = $this->evaluate($evalEpisodes);
                $history['step'][]=$step; $history['episodes'][]=$episodeCount; $history['evalReward'][]=$eval;
                $marker = $eval>$bestEval ? ' <- best' : '';
                if ($eval>$bestEval) {
                    $bestEval=$eval;
                    if ($bestModelFile!==null) $this->agent->saveWeightsToFile($bestModelFile);
                }
                $progress->clearProgressBar();
                printf("Step %7d | EvalDet=%+8.2f | Episodes=%d%s\n",$step,$eval,$episodeCount,$marker);
                if ($this->solvedReward!==null && $eval >= $this->solvedReward) {
                    echo "Solved! (deterministic mean reward >= {$this->solvedReward})\n"; break;
                }
            }
        }
        echo "\nTraining finished. Best eval reward: {$bestEval} time: {$progress->laptimeString()}\n";
        return $history;
    }
}
