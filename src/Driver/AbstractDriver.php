<?php
namespace Rindow\RL\Agents\Driver;

use Interop\Polite\AI\RL\Environment as Env;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Driver;
use Rindow\RL\Agents\ReplayBuffer as ReplayBufferInterface;
use Rindow\RL\Agents\EventManager as EventManagerInterface;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Util\EventManager;

abstract class AbstractDriver implements Driver
{
    protected $la;
    protected $agent;
    protected $experience;
    protected $eventManager;
    protected $experienceSize;
    protected $customRewardFunction;
    protected $customObservationFunction;

    public function __construct(
        object $la, Agent $agent, int $experienceSize,
        ReplayBufferInterface $replayBuffer=null,
        EventManagerInterface $eventManager=null)
    {
        $this->la = $la;
        $this->agent = $agent;
        $this->experienceSize = $experienceSize;
        if($replayBuffer===null) {
            $replayBuffer = new ReplayBuffer($this->la,$this->experienceSize);
        }
        $this->experience = $replayBuffer;
        if($eventManager===null) {
            $eventManager = new EventManager();
        }
        $this->eventManager = $eventManager;
        $agent->register($eventManager);
    }

    protected function onStartEpisode() : void
    {
        $this->eventManager->notify(Driver::EVENT_START_EPISODE);
    }

    protected function onEndEpisode() : void
    {
        $this->eventManager->notify(Driver::EVENT_END_EPISODE);
    }

    public function setCustomRewardFunction(callable $func) : void
    {
        $this->customRewardFunction = $func;
    }

    public function setCustomObservationFunction(callable $func) : void
    {
        $this->customObservationFunction = $func;
    }

    protected function customReward($env,$stepCount,$observation,$reward,$done,$info) : float
    {
        $func = $this->customRewardFunction;
        if($func===null) {
            return $reward;
        }
        return $func($env,$stepCount,$observation,$reward,$done,$info);
    }

    protected function customObservation($env,$observation,$done)
    {
        $func = $this->customObservationFunction;
        if($func===null) {
            return $observation;
        }
        return $func($env,$observation,$done);
    }

    protected function initialize()
    {
        $this->experience->clear();
    }

    public function agent() : Agent
    {
        return $this->agent;
    }

    protected function console($message)
    {
        if(defined('STDERR')) {
            fwrite(STDERR,$message);
        }
    }

    public function evaluation(Env $env,int $episodes, array $metrics) : array
    {
        $agent = $this->agent;
        $sumReward = $sumSteps = 0;
        for($episode=0;$episode<$episodes;$episode++) {
            $observation = $env->reset();
            $observation = $this->customObservation($env,$observation,false);
            $done = false;
            $episodeSteps = 0;
            while(!$done) {
                $action = $agent->action($observation,$training=false);
                [$observation,$reward,$done,$info] = $env->step($action);
                $observation = $this->customObservation($env,$observation,$done);
                $reward = $this->customReward($env,$episodeSteps,$observation,$reward,$done,$info);
                $sumReward += $reward;
                $episodeSteps++;
            }
            $sumSteps += $episodeSteps;
        }
        $report = [];
        $report['val_steps'] = $sumSteps/$episodes;
        $report['val_reward'] = $sumReward/$episodes;
        return $report;
    }

    protected function progressBar($title,$iterNumber,$numIterations,$evalInterval,$startTime,$maxDot)
    {
        if($iterNumber<0) {
            $this->console("\r${title} 1/${numIterations} ");
            return;
        }
        $iterNumber++;
        $elapsed = time() - $startTime;
        if($evalInterval) {
            $completion = $iterNumber / $numIterations;
            $progressOfAgg = ((($iterNumber-1)%$evalInterval)+1) / $evalInterval;
            $estimated = $elapsed / $completion;
            $remaining = $estimated - $elapsed;
            $dot = (int)ceil($maxDot*$progressOfAgg);
            $sec = (int)floor($remaining) % 60;
            $min = (int)floor($remaining/60) % 60;
            $hour = (int)floor($remaining/3600);
            $rem_string = ($hour?$hour.':':'').sprintf('%02d:%02d',$min,$sec);
        } else {
            $dot = 1;
            $rem_string = '????';
            $this->console($maxDot."\n");
        }
        $this->console("\r${title} ${iterNumber}/${numIterations} [".
            str_repeat('.',$dot).str_repeat(' ',$maxDot-$dot).
            "] ${elapsed} sec. remaining:${rem_string}  ");
    }
}
