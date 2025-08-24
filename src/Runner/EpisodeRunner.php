<?php
namespace Rindow\RL\Agents\Runner;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Runner;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\ReplayBuffer;
use InvalidArgumentException;

class EpisodeRunner extends AbstractRunner
{
    protected $env;
    protected ?Env $evalEnv;

    public function __construct(
        object $la,
        Env $env,
        Agent $agent,
        ?int $experienceSize=null,
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
        ?int $targetScore=null, ?int $numAchievements=null,
        ?int $verbose=null
    ) : array
    {
        $numIterations ??= 1000;
        // $numRolloutSteps : N/A
        $evalInterval ??= 100;
        $numEvalEpisodes ??= 10;
        $logInterval ??= 10;
        $verbose ??= 0;
        $env = $this->env;
        $agent = $this->agent;
        $metrics ??= [];
        if($numEvalEpisodes!=0) {
            if($this->evalEnv===null) {
                throw new InvalidArgumentException("You cannot specify `numEvalEpisodes` without an `evalEnv` being specified.");
            }
        }
        $this->metrics()->attract($metrics);
        $metrics = $this->metrics();
        $numAchievements ??= 5;

        $history = [];
        $isStepUpdate = $agent->isStepUpdate();
        $subStepLen = $agent->subStepLength();
        $totalStep = 0;
        $startTime = time();
        if($verbose>=2) {
            $logStartTime = microtime(true);
            $logStepCount = 0;
        }

        $epsilonMetric = false;
        if($metrics->isAttracted('epsilon') &&
            method_exists($agent,'policy') ) {
            $policy = $agent->policy();
            if($policy && method_exists($policy,'getEpsilon')) {
                $epsilonMetric = true;
            }
        }

        if($verbose>=1) {
            $message = "Train on {$numIterations} episodes with {$numEvalEpisodes} evaluation each aggregation.\n";
            $this->console($message);
            $message = "Eval-Interval:$evalInterval Refresh:$logInterval\n";
            $this->console($message);
            $this->progressBar('Episode',0,$numIterations,$startTime,25);
        }
        $episode=0;
        while($episode<$numIterations) {
            $experience = $this->experience;
            $episodeReward = 0.0;
            $episodeSteps = 0;
            $episodeLoss = 0.0;
            $done = false;
            $truncated = false;
            //[$states,$info] = $env->reset();
            //$states = $this->customState($env,$states,$done,$truncated,$info);
            [$states,$info] = $agent->reset($env);
            while(!($done || $truncated)) {
                if($maxSteps!==null) {
                    if($episodeSteps>=$maxSteps) {
                        break;
                    }
                }
                //$action = $agent->action($states,training:true,info:$info);
                //[$nextState,$reward,$done,$truncated,$info] = $env->step($action);
                //$nextState = $this->customState($env,$nextState,$done,$truncated,$info);
                //$reward = $this->customReward($env,$episodeSteps,$states,$action,$nextState,$reward,$done,$truncated,$info);
                //$experience->add([$states,$action,$nextState,$reward,$done,$truncated,$info]);
                [$nextState,$reward,$done,$truncated,$info] = $agent->collect($env,$experience,$episodeSteps,$states,info:$info);
                $totalStep++;
                if($agent->isStepUpdate() && $totalStep>=$subStepLen) {
                    $agent->update($experience);
                }
                $states = $nextState;
                $episodeReward += $reward;
                $episodeSteps++;
                if($verbose>=2) {
                    $logStepCount++;
                }
            }

            if(!$agent->isStepUpdate()) {
                $agent->update($experience);
            }
            $metrics->update('reward',$episodeReward);
            $metrics->update('steps',$episodeSteps);
            $metrics->update('logreward',$episodeReward);
            $metrics->update('logsteps',$episodeSteps);
            $this->onEndEpisode();
            $episode++;
            $episodeReward = 0.0;
            $episodeSteps = 0;

            if(($episode)%$logInterval==0 || ($episode)%$evalInterval==0) {
                if($epsilonMetric) {
                    $epsilon = $policy->getEpsilon();
                    $metrics->update('epsilon',$epsilon);
                }
            }
            if(($episode)%$logInterval==0) {
                if($verbose>=2) {
                    $logText = $metrics->render(exclude:['steps','reward','valSteps','valRewards']);
                    //$qLog = sprintf('%1.1f',$agent->getQValue($states));
                    $msPerStep = sprintf('%1.1f',($logInterval>0)?((microtime(true) - $logStartTime)/$logStepCount*1000):0);
                    $logsteps = $metrics->printable('logsteps');
                    $logreward = $metrics->printable('logreward');
                    $metrics->reset('logsteps');
                    $metrics->reset('logreward');
                    //$this->console("Step:".($step)." Ep:".($episode)." rw={$rewardLog}, st={$stepsLog} loss={$lossLog}{$epsilonLog}, q={$qLog}, {$msPerStep}ms/st\n");
                    $this->clearProgressBar();
                    $this->console("Episode:{$episode} St:{$totalStep} {$logsteps} {$logreward} {$logText} {$msPerStep}ms/st\n");
                }
                if($verbose>=1) {
                    $this->progressBar('Episode',$episode,$numIterations,$startTime,25);
                }
            }
            if($episode%$evalInterval == 0) {
                if($numEvalEpisodes!=0) {
                    $evalReport = $this->evaluation($this->evalEnv,$experience,$numEvalEpisodes);
                    $metrics->update('valSteps',$evalReport['valSteps']);
                    $metrics->update('valRewards',$evalReport['valRewards']);
                }
                if($verbose>=1) {
                    $logText = $metrics->render();
                    $this->clearProgressBar();
                    $this->console("Episode:{$episode} St:{$totalStep} {$logText}\n");
                    $this->progressBar('Episode',$episode,$numIterations,$startTime,25);
                }
                $metrics->update('iter',$episode);
                $metrics->record();
                if($targetScore!==null) {
                    if($metrics->result('reward')>$targetScore) {
                        $countAchievements += 1;
                        if($countAchievements>=$numAchievements) {
                            if($verbose>=1) {
                                $this->console("\n");
                                $this->console("I have achieved my goal score of {$targetScore}. I will stop training.");
                            }
                            break;
                        }
                    } else {
                        $countAchievements = 0;
                    }
                }
                $metrics->resetAll();
            }
            if($episode%$logInterval==0) {
                $logStartTime = microtime(true);
                $logStepCount = 0;
            }
        }
        if($verbose>=1) {
            $this->console("\n");
        }
        return $metrics->history();
    }
}
