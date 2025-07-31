<?php
namespace Rindow\RL\Agents\Runner;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Runner;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\ReplayBuffer;

class StepRunner extends AbstractRunner
{
    protected Env $env;
    protected ?Env $evalEnv;

    public function __construct(
        object $la,
        Env $env,
        Agent $agent,
        int $experienceSize,
        ?ReplayBuffer $replayBuffer=null,
        ?Env $evalEnv=null,
        )
    {
        parent::__construct($la,$agent,$experienceSize,$replayBuffer);
        $this->env = $env;
        $this->evalEnv = $evalEnv;
        $this->initialize();
    }

    public function train(
        ?int $numIterations=null, ?int $numRolloutSteps=null, ?int $maxSteps=null, ?array $metrics=null,
        ?int $evalInterval=null, ?int $numEvalEpisodes=null, ?int $logInterval=null,
        ?int $verbose=null) : array
    {
        $numIterations ??= 1000;
        $evalInterval ??= 100;
        $numEvalEpisodes ??= 0;
        $logInterval ??= 100;
        $verbose ??= 0;
        $env = $this->env;
        $agent = $this->agent;
        $metrics ??= [];

        $numRolloutSteps = $this->agent->numRolloutSteps();
        $numIterations = max($numIterations,$numRolloutSteps);
        $evalInterval = max($evalInterval,$numRolloutSteps);
        $logInterval = max($logInterval,$numRolloutSteps);
        $evalInterval = $evalInterval - ($evalInterval % $numRolloutSteps);
        $logInterval = $logInterval - ($logInterval % $numRolloutSteps);


        $this->metrics->attract($metrics);
        $metrics = $this->metrics;
        $isStepUpdate = $agent->isStepUpdate();
        $subStepLen = $agent->subStepLength();
        $startTime = time();
        $episode = 0;
        $episodeCount = $sumReward = $sumSteps = $sumLoss = $countLoss = 0;
        if($verbose>0) {
            $this->console("Train on {$numIterations} steps with {$numEvalEpisodes} evaluation each aggregation.\n");
        }
        // verbose=2 log
        $logStartTime = microtime(true);
        $logCountLoss = $logSumLoss = $logEpisodeCount = 0;

        // start episode
        $this->onStartEpisode();
        [$states,$info] = $env->reset();
        $states = $this->customState($env,$states,false,false,$info);
        $experience = $this->experience;
        $episodeReward = $reward = 0.0;
        $episodeSteps = 0;
        $done = false;
        $truncated = false;

        $epsilonMetric = false;
        if($metrics->isAttracted('epsilon') &&
            method_exists($agent,'policy') ) {
            $policy = $agent->policy();
            if($policy && method_exists($policy,'getEpsilon')) {
                $epsilonMetric = true;
            }
        }

        if($verbose>=1) {
            $this->progressBar('Step',0,$numIterations,$startTime,25);
        }
        $step = 0;
        while($step<$numIterations) {
            $rollout = 0;
            while($rollout<$numRolloutSteps && $step<$numIterations) {
                $action = $agent->action($states,training:true,info:$info);
                [$nextState,$reward,$done,$truncated,$info] = $env->step($action);
                $nextState = $this->customState($env,$nextState,$done,$truncated,$info);
                $reward = $this->customReward($env,$episodeSteps,$states,$action,$nextState,$reward,$done,$truncated,$info);
                $experience->add([$states,$action,$nextState,$reward,$done,$truncated,$info]);
                $states = $nextState;
                $episodeReward += $reward;
                $episodeSteps++;
                $rollout++;
                $step++;

                // End Episode
                if($done || $truncated) {
                    $metrics->update('reward',$episodeReward);
                    $metrics->update('steps',$episodeSteps);
                    $this->onEndEpisode();
                    $episode++;
                    $episodeReward = 0.0;
                    $episodeSteps = 0;
                    $this->onStartEpisode();
                    // start episode
                    $done = false;
                    $truncated = false;
                    [$states,$info] = $env->reset();
                    $states = $this->customState($env,$states,false,false,$info);
                    $experience = $this->experience;
                }

            }
            if($step<$subStepLen) {
                continue;
            }
            $agent->update($experience);
            // Update Progress bar and Logging metrics for short time.
            if(($step)%$logInterval==0) {
                if($epsilonMetric) {
                    $epsilon = $policy->getEpsilon();
                    $metrics->update('epsilon',$epsilon);
                }
                if($verbose>1) {
                    $logText = $metrics->render(exclude:['valSteps','valRewards']);
                    //$qLog = sprintf('%1.1f',$agent->getQValue($states));
                    $msPerStep = sprintf('%1.1f',($logInterval>0)?((microtime(true) - $logStartTime)/$logInterval*1000):0);
                    //$this->console("Step:".($step)." Ep:".($episode)." rw={$rewardLog}, st={$stepsLog} loss={$lossLog}{$epsilonLog}, q={$qLog}, {$msPerStep}ms/st\n");
                    $this->clearProgressBar();
                    $this->console("Step:".($step)." Ep:".($episode)." {$logText} {$msPerStep}ms/st\n");
                }
                if($verbose>0) {
                    $this->progressBar('Step',$step,$numIterations,$startTime,25);
                }
            }
            // Evaluation and Logging Metrics
            if(($step)%$evalInterval==0) {
                if($numEvalEpisodes!=0) {
                    $evalReport = $this->evaluation($this->evalEnv,$numEvalEpisodes);
                    if($verbose>0) {
                        $metrics->update('valSteps',$evalReport['valSteps']);
                        $metrics->update('valRewards',$evalReport['valRewards']);
                        $logText = $metrics->render();
                        $this->clearProgressBar();
                        $this->console("Step:".($step)." Ep:".($episode)." $logText\n");
                        $this->progressBar('Step',$step,$numIterations,$startTime,25);
                    }
                }
                $metrics->record();
                $metrics->resetAll();
            }
            if(($step)%$logInterval==0) {
                $logStartTime = microtime(true);
            }
            if($maxSteps!==null) {
                if($episodeSteps>=$maxSteps) {
                    //$done=true;
                    $truncated = true;
                }
            }
        }
        return $metrics->history();
    }
}
