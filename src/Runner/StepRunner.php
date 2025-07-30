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
        ?int $numIterations=null, ?int $maxSteps=null, ?array $metrics=null,
        ?int $evalInterval=null, ?int $numEvalEpisodes=null, ?int $logInterval=null,
        ?int $verbose=null) : array
    {
        if($numIterations===null || $numIterations<=0) {
            $numIterations = 1000;
        }
        if($evalInterval===null || $evalInterval<=0) {
            $evalInterval = 100;
        }
        if($numEvalEpisodes===null) {
            $numEvalEpisodes = 10;
        } elseif($numEvalEpisodes<0) {
            $numEvalEpisodes = 0;
        }
        if($logInterval===null || $logInterval<=0) {
            $logInterval = 100;
        }
        if($verbose===null) {
            $verbose = 0;
        }
        $env = $this->env;
        $agent = $this->agent;
        if($metrics===null) {
            $metrics = [];
        }
        $history = $this->history;
        $history->attract($metrics);

        $isStepUpdate = $agent->isStepUpdate();
        $subStepLen = $agent->subStepLength();
        $totalStep = 0;
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
        if($history->isAttracted('epsilon') &&
            method_exists($agent,'policy') ) {
            $policy = $agent->policy();
            if($policy && method_exists($policy,'getEpsilon')) {
                $epsilonMetric = true;
            }
        }

        if($verbose>=1) {
            $this->progressBar('Step',0,$numIterations,$startTime,25);
        }
        for($step=0;$step<$numIterations;$step++) {
            $action = $agent->action($states,training:true,info:$info);
            [$nextState,$reward,$done,$truncated,$info] = $env->step($action);
            $nextState = $this->customState($env,$nextState,$done,$truncated,$info);
            $reward = $this->customReward($env,$episodeSteps,$states,$action,$nextState,$reward,$done,$truncated,$info);
            $experience->add([$states,$action,$nextState,$reward,$done,$truncated,$info]);
            $totalStep++;
            if($agent->isStepUpdate() && $totalStep>=$subStepLen) {
                $loss = $agent->update($experience);
            }
            $states = $nextState;
            $episodeReward += $reward;
            $episodeSteps++;

            if($done || $truncated) {
                if(!$agent->isStepUpdate()) {
                    $loss = $agent->update($experience);
                }
                $this->history->update('reward',$episodeReward);
                $this->history->update('steps',$episodeSteps);
                $this->onEndEpisode();
            }
            // Update Progress bar and Logging metrics for short time.
            if(($step+1)%$logInterval==0) {
                if($epsilonMetric) {
                    $epsilon = $policy->getEpsilon();
                    $history->update('epsilon',$epsilon);
                }
                if($verbose>1) {
                    $logText = $history->render(exclude:['valSteps','valReward']);
                    //$qLog = sprintf('%1.1f',$agent->getQValue($states));
                    $msPerStep = sprintf('%1.1f',($logInterval>0)?((microtime(true) - $logStartTime)/$logInterval*1000):0);
                    //$this->console("Step:".($step+1)." ep:".($episode+1)." rw={$rewardLog}, st={$stepsLog} loss={$lossLog}{$epsilonLog}, q={$qLog}, {$msPerStep}ms/st\n");
                    $this->clearProgressBar();
                    $this->console("Step:".($step+1)." ep:".($episode+1)." {$logText} {$msPerStep}ms/st\n");
                }
                if($verbose>0) {
                    $this->progressBar('Step',$step,$numIterations,$startTime,25);
                }
            }
            // Evaluation and Logging Metrics
            if(($step+1)%$evalInterval==0) {
                if($numEvalEpisodes!=0) {
                    $evalReport = $this->evaluation($this->evalEnv,$numEvalEpisodes,$metrics);
                    if($verbose>0) {
                        $this->history->update('valSteps',$evalReport['valSteps']);
                        $this->history->update('valReward',$evalReport['valReward']);
                        $logText = $history->render();
                        $this->clearProgressBar();
                        $this->console("Step:".($step+1)." ep:".($episode+1)." $logText\n");
                        $this->progressBar('Step',$step,$numIterations,$startTime,25);
                    }
                }
                $history->record();
                $history->resetAll();
            }
            if(($step+1)%$logInterval==0) {
                $logStartTime = microtime(true);
            }
            if($maxSteps!==null) {
                if($episodeSteps>=$maxSteps) {
                    //$done=true;
                    $truncated = true;
                }
            }
            if($done||$truncated) {
                // start episode
                $episode++;
                $this->onStartEpisode();
                [$states,$info] = $env->reset();
                $states = $this->customState($env,$states,false,false,$info);
                $experience = $this->experience;
                $episodeReward = 0.0;
                $episodeSteps = 0;
                $done = false;
                $truncated = false;
            }
        }
        return $history->history();
    }
}
