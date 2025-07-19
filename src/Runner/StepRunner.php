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
        $history = [];
        if($metrics===null) {
            $metrics = [];
        }
        foreach($metrics as $key) {
            $history[$key] = [];
        }
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

        for($step=0;$step<$numIterations;$step++) {
            if($verbose==1&&$step==0) {
                $this->progressBar('Step',$step,$numIterations,$evalInterval,$startTime,25);
            }
            $action = $agent->action($states,training:true,info:$info);
            [$nextState,$reward,$done,$truncated,$info] = $env->step($action);
            $nextState = $this->customState($env,$nextState,$done,$truncated,$info);
            $reward = $this->customReward($env,$episodeSteps,$states,$action,$nextState,$reward,$done,$truncated,$info);
            $experience->add([$states,$action,$nextState,$reward,$done,$truncated,$info]);
            $totalStep++;
            if($agent->isStepUpdate() && $totalStep>=$subStepLen) {
                $loss = $agent->update($experience);
                if($loss!==null) {
                    $sumLoss += $loss;
                    $logSumLoss += $loss;
                }
                $countLoss++;
                $logCountLoss++;
            }
            $states = $nextState;
            $episodeReward += $reward;
            $episodeSteps++;

            if($done || $truncated) {
                if(!$agent->isStepUpdate()) {
                    $loss = $agent->update($experience);
                    if($loss!==null) {
                        $sumLoss += $loss;
                        $logSumLoss += $loss;
                    }
                    $countLoss++;
                    $logCountLoss++;
                }
                $sumReward += $episodeReward;
                $sumSteps += $episodeSteps;
                $this->onEndEpisode();
                $episodeCount++;
                $logEpisodeCount++;
            }
            $epsilon = null;
            if(($step+1)%$logInterval==0 || ($step+1)%$evalInterval==0) {
                if(in_array('epsilon',$metrics)) {
                    if(method_exists($agent,'policy')) {
                        $policy = $agent->policy();
                        if($policy && method_exists($policy,'getEpsilon')) {
                            $epsilon = $policy->getEpsilon();
                        }
                    }
                }
            }
            if(($step+1)%$logInterval==0) {
                if($epsilon!==null) {
                    $epsilonLog = ', eps='.sprintf('%5.3f',$epsilon);
                } else {
                    $epsilonLog = '';
                }
                if($verbose>1) {
                    $stepsLog = sprintf('%1.1f',($logEpisodeCount>0)? ($logInterval/$logEpisodeCount) : 0);
                    $rewardLog = sprintf('%1.1f',($logEpisodeCount>0)? ($episodeReward/$logEpisodeCount) : 0);
                    $lossLog = sprintf('%3.2e',($logCountLoss>0)?($logSumLoss/$logCountLoss):0);
                    //$qLog = sprintf('%1.1f',$agent->getQValue($states));
                    $msPerStep = sprintf('%1.1f',($logInterval>0)?((microtime(true) - $logStartTime)/$logInterval*1000):0);
                    //$this->console("Step:".($step+1)." ep:".($episode+1)." rw={$rewardLog}, st={$stepsLog} loss={$lossLog}{$epsilonLog}, q={$qLog}, {$msPerStep}ms/st\n");
                    $this->console("Step:".($step+1)." ep:".($episode+1)." rw={$rewardLog}, st={$stepsLog} loss={$lossLog}{$epsilonLog}, {$msPerStep}ms/st\n");
                } elseif($verbose==1) {
                    $this->progressBar('Step',$step,$numIterations,$evalInterval,$startTime,25);
                }
                $logEpisodeCount = $logSumLoss = $logCountLoss = 0;
            }
            if(($step+1)%$evalInterval==0) {
                if($numEvalEpisodes!=0) {
                    $evalReport = $this->evaluation($this->evalEnv,$numEvalEpisodes,$metrics);
                } else {
                    $evalReport = [];
                }
                if($epsilon!==null && in_array('epsilon',$metrics)) {
                    $history['epsilon'][] = $epsilon;
                }
                $avgSteps = ($episodeCount>0)? ($sumSteps/$episodeCount) : 0;
                $avgReward = ($episodeCount>0)? ($sumReward/$episodeCount) : 0;
                $avgLoss = ($countLoss>0)? ($sumLoss/$countLoss) : 0;
                if(in_array('steps',$metrics)) {
                    $history['steps'][] = $avgSteps;
                }
                if(in_array('reward',$metrics)) {
                    $history['reward'][] = $avgReward;
                }
                if(in_array('loss',$metrics)) {
                    $history['loss'][] = $avgLoss;
                }
                foreach ($evalReport as $key => $value) {
                    if(in_array($key, $metrics)) {
                        $history[$key][] = $value;
                    }
                }
                if($verbose>0) {
                    if($epsilon!==null) {
                        $epsilon = ', eps='.sprintf('%5.3f',$epsilon);
                    } else {
                        $epsilon = '';
                    }
                    $avgSteps = sprintf('%3.1f',$avgSteps);
                    $avgReward = sprintf('%3.2f',$avgReward);
                    if($avgLoss!=0) {
                        $avgLoss = ', Loss='.sprintf('%3.2e',$avgLoss);
                    } else {
                        $avgLoss = '';
                    }
                    if($numEvalEpisodes>0) {
                        $valSteps = sprintf('%3.1f',$evalReport['val_steps']);
                        $valReward = sprintf('%3.2f',$evalReport['val_reward']);
                    } else {
                        $valSteps = '-';
                        $valReward = '-';
                    }
                    if($verbose==1) {
                        $this->console("\n");
                    }
                    $this->console("Avg Rwd={$avgReward}, St={$avgSteps}{$avgLoss},".
                                    " vRwd={$valReward}, vSt={$valSteps}{$epsilon}\n");
                }
                $episodeCount = 0;
                $sumSteps = 0;
                $sumReward = 0;
                $sumLoss = 0;
                $countLoss = 0;
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
                $episodeReward = $reward = 0.0;
                $episodeSteps = 0;
                $episodeLoss = 0.0;
                $done = false;
                $truncated = false;
            }
        }
        return $history;
    }
}
