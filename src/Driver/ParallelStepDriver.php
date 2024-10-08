<?php
namespace Rindow\RL\Agents\Driver;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Driver;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

class ParallelStepDriver extends AbstractDriver
{
    protected $envs;
    protected $evalEnv;
    protected $experiences;

    public function __construct(object $la,
        array $envs, Agent $agent, int $experienceSize, array $replayBuffers=null,
        Env $evalEnv=null)
    {
        parent::__construct($la,$agent,$experienceSize);
        $this->envs = $envs;
        $this->evalEnv = $evalEnv;
        $numEnvs = count($envs);
        if($replayBuffers===null) {
            for($i=0;$i<$numEnvs;$i++) {
                $replayBuffers[] = new ReplayBuffer($this->la,$experienceSize);
            }
        }
        $this->experiences = $replayBuffers;
        $this->initialize();
    }

    public function train(
        int $numIterations=null, int $maxSteps=null, array $metrics=null,
        int $evalInterval=null, int $numEvalEpisodes=null, int $logInterval=null,
        int $verbose=null) : array
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
        $envs = $this->envs;
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
        $startTime = time();
        $stepCount = $episodeCount = $sumReward = $sumLoss = $countLoss = 0;
        $numEnvs = count($envs);
        if($verbose>0) {
            $this->console("Train on {$numIterations} steps with {$numEvalEpisodes} evaluation each aggregation.\n");
        }
        // verbose=2 log
        $logStartTime = microtime(true);
        $logCountLoss = $logSumReward = $logSumLoss = $logStepCount = $logEpisodeCount = 0;
        $totalEpisodeCount = 0;

        // start episode
        $obss = [];
        $episodeSteps = [];
        foreach($envs as $env) {
            $observation = $env->reset();
            $obss[] = $this->customObservation($env,$observation,false);
            $episodeSteps[] = 0;
        }
        $experiences = $this->experiences;

        for($step=0;$step<$numIterations;$step++) {
            $nextObss = [];
            if($verbose==1&&$step==0) {
                $this->progressBar('Step',$step,$numIterations,$evalInterval,$startTime,25);
            }
            $actions = $agent->action($obss,$training=true);
            foreach($envs as $i => $env) {
                [$nextObs,$reward,$done,$info] = $env->step($actions[$i]);
                $nextObss[$i] = $this->customObservation($env,$nextObs,$done);
                $reward = $this->customReward($env,$episodeSteps[$i],$nextObs,$reward,$done,$info);
                $experiences[$i]->add([$obss[$i],$actions[$i],$nextObss[$i],$reward,$done,$info]);
                $loss = $agent->update($experiences[$i]);
                if($loss!==null) {
                    $sumLoss += $loss;
                    $logSumLoss += $loss;
                    $countLoss++;
                    $logCountLoss++;
                }
                $sumReward += $reward;
                $logSumReward += $reward;
                $stepCount++;
                $logStepCount++;
                $episodeSteps[$i]++;
                if($done || ($maxSteps!==null && $episodeSteps[$i]>=$maxSteps)) {
                    // start episode
                    $totalEpisodeCount++;
                    $episodeCount++;
                    $logEpisodeCount++;
                    $nextObs = $env->reset();
                    $nextObss[$i] = $this->customObservation($env,$nextObs,false);
                    $episodeSteps[$i] = 0;
                }
            }
            $obss = $nextObss;

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
                    $stepLog = sprintf('%d',($step+1)*$numEnvs);
                    $episodeLog = sprintf('%d',$totalEpisodeCount);
                    $stepsLog = sprintf('%1.1f',($logEpisodeCount>0)? ($logStepCount/$logEpisodeCount) : 0);
                    $rewardLog = sprintf('%1.1f',($logEpisodeCount>0)? ($logSumReward/$logEpisodeCount) : 0);
                    $lossLog = sprintf('%3.2e',($logCountLoss>0)?($logSumLoss/$logCountLoss):0);
                    //$qLog = sprintf('%1.1f',$agent->getQValue($observation));
                    $msPerStep = sprintf('%1.1f',($logInterval>0)?((microtime(true) - $logStartTime)/$logInterval*1000):0);
                    //$this->console("Step:{$stepLog} ep:{$episodeLog} rw={$rewardLog}, st={$stepsLog} loss={$lossLog}{$epsilonLog}, q={$qLog}, {$msPerStep}ms/st\n");
                    $this->console("Step:{$stepLog} ep:{$episodeLog} rw={$rewardLog}, st={$stepsLog} loss={$lossLog}{$epsilonLog}, {$msPerStep}ms/st\n");
                } elseif($verbose==1) {
                    $this->progressBar('Step',$step,$numIterations,$evalInterval,$startTime,25);
                }
                $logEpisodeCount = $logSumLoss = $logCountLoss = $logStepCount = $logSumReward = 0;
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
                $avgSteps = ($episodeCount>0)? ($stepCount/$episodeCount) : 0;
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
                $stepCount = 0;
                $sumReward = 0;
                $sumLoss = 0;
                $countLoss = 0;
            }
            if(($step+1)%$logInterval==0) {
                $logStartTime = microtime(true);
            }
        }
        return $history;
    }
}
