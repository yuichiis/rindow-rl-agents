<?php
namespace Rindow\RL\Agents\Runner;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Runner;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\ReplayBuffer;

class EpisodeRunner extends AbstractRunner
{
    protected $env;

    public function __construct(
        object $la,
        Env $env,
        Agent $agent,
        int $experienceSize,
        ?ReplayBuffer $replayBuffer=null,
        )
    {
        parent::__construct($la,$agent,$experienceSize,$replayBuffer);
        $this->env = $env;
        $this->initialize();
    }

    public function train(
        ?int $numIterations=null,?int $maxSteps=null,?array $metrics=null,
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
        // $logInterval :  N/A
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
        $epStartTime = 0.0;
        $episodeCount = $sumReward = $sumSteps = $sumLoss = $countLoss = 0;
        if($verbose>0) {
            $this->console("Train on {$numIterations} episodes with {$numEvalEpisodes} evaluation each aggregation.\n");
        }
        for($episode=0;$episode<$numIterations;$episode++) {
            if($verbose==1&&$episode==0) {
                $this->progressBar('Episode',$episode,$numIterations,$startTime,25);
            }
            if($verbose>1) {
                $epStartTime = microtime(true);
            }
            $experience = $this->experience;
            $episodeReward = 0.0;
            $episodeSteps = 0;
            $episodeLoss = 0.0;
            $done = false;
            $truncated = false;
            [$states,$info] = $env->reset();
            $states = $this->customState($env,$states,$done,$truncated,$info);
            while(!($done || $truncated)) {
                if($maxSteps!==null) {
                    if($episodeSteps>=$maxSteps) {
                        break;
                    }
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
                        $episodeLoss += $loss;
                    }
                    $countLoss++;
                }
                $states = $nextState;
                $episodeReward += $reward;
                $episodeSteps++;
            }
            if(!$agent->isStepUpdate()) {
                $loss = $agent->update($experience);
                if($loss!==null) {
                    $sumLoss += $loss;
                    $episodeLoss += $loss;
                }
                $countLoss++;
            }
            $sumReward += $episodeReward;
            $sumSteps += $episodeSteps;
            $this->onEndEpisode();
            $episodeCount++;
            $epsilon = null;
            if($verbose>1 ||
               ($episodeCount >= $evalInterval && in_array('epsilon',$metrics))) {
                if(method_exists($agent,'policy')) {
                    $policy = $agent->policy();
                    if($policy && method_exists($policy,'getEpsilon')) {
                        $epsilon = $policy->getEpsilon();
                    }
                }
            }
            if($episodeCount >= $evalInterval) {
                if($numEvalEpisodes!=0) {
                    $evalReport = $this->evaluation($this->env, $numEvalEpisodes,$metrics);
                } else {
                    $evalReport = [];
                }
                if($epsilon!==null && in_array('epsilon',$metrics)) {
                    $history['epsilon'][] = $epsilon;
                }
                $avgSteps = $sumSteps/$episodeCount;
                $avgReward = $sumReward/$episodeCount;
                if($countLoss!=0 && $sumLoss!=0) {
                    $avgLoss = $sumLoss/$countLoss;
                } else {
                    $avgLoss = 0;
                }
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
            }
            if($epsilon!==null) {
                $epsilon = ', eps='.sprintf('%5.3f',$epsilon);
            } else {
                $epsilon = '';
            }
            if($verbose>1) {
                $strEpisodeReward =  sprintf('%1.2f',$episodeReward);
                $lossLog = sprintf('%3.2e',$episodeLoss/$episodeSteps);
                //$qLog = sprintf('%1.1f',$agent->getQValue($states));
                $msPerStep = sprintf('%1.1f',(microtime(true) - $epStartTime)/$episodeSteps*1000);
                //$this->console("Ep ".($episode+1).": rw={$strEpisodeReward}, st={$episodeSteps} loss={$lossLog}{$epsilon}, q={$qLog}, {$msPerStep}ms/step\n");
                $this->console("Ep ".($episode+1).": rw={$strEpisodeReward}, st={$episodeSteps} loss={$lossLog}{$epsilon}, {$msPerStep}ms/step\n");
            } elseif($verbose==1) {
                $this->progressBar('Episode',$episode,$numIterations,$startTime,25);
            }
            if($verbose>0) {
                if($episodeCount >= $evalInterval) {
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
            }
            if($episodeCount >= $evalInterval) {
                $episodeCount = 0;
                $sumSteps = 0;
                $sumReward = 0;
                $sumLoss = 0;
                $countLoss = 0;
            }
        }
        return $history;
    }
}
