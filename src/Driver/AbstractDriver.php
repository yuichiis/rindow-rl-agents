<?php
namespace Rindow\RL\Agents\Driver;

use Interop\Polite\AI\RL\Environment as Env;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent;
use Rindow\RL\Agents\Driver;
use Rindow\RL\Agents\ReplayBuffer as ReplayBufferInterface;
use Rindow\RL\Agents\EventManager as EventManagerInterface;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Util\EventManager;

abstract class AbstractDriver implements Driver
{
    protected object $la;
    protected Agent $agent;
    protected ReplayBufferInterface $experience;
    protected EventManagerInterface $eventManager;
    protected int $experienceSize;
    protected mixed $customRewardFunction=null;
    protected mixed $customStateFunction=null;

    public function __construct(
        object $la, Agent $agent, int $experienceSize,
        ?ReplayBufferInterface $replayBuffer=null,
        ?EventManagerInterface $eventManager=null)
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

    public function setCustomStateFunction(callable $func) : void
    {
        $this->customStateFunction = $func;
    }

    protected function customReward(
        Env $env,
        int $stepCount,
        NDArray $states,
        float $reward,
        bool $done,
        bool $truncated,
        ?array $info,
        ) : float
    {
        $func = $this->customRewardFunction;
        if($func===null) {
            return $reward;
        }
        return $func($env,$stepCount,$states,$reward,$done,$truncated,$info);
    }

    protected function customState(
        Env $env,
        NDArray $states,
        bool $done,
        bool $truncated,
        ?array $info,
        ) : NDArray
    {
        $func = $this->customStateFunction;
        if($func===null) {
            return $states;
        }
        return $func($env,$states,$done,$truncated,$info);
    }

    protected function initialize() : void
    {
        $this->experience->clear();
    }

    public function agent() : Agent
    {
        return $this->agent;
    }

    protected function console(string $message)
    {
        if(defined('STDERR')) {
            fwrite(STDERR,$message);
        }
    }

    public function evaluation(
        Env $env,
        int $episodes,
        array $metrics
        ) : array
    {
        $agent = $this->agent;
        $sumReward = $sumSteps = 0;
        for($episode=0;$episode<$episodes;$episode++) {
            [$states,$info] = $env->reset();
            $states = $this->customState($env,$states,false,false,$info);
            $done = false;
            $truncated = false;
            $episodeSteps = 0;
            while(!($done || $truncated)) {
                $action = $agent->action($states,training:false,info:$info);
                [$states,$reward,$done,$truncated,$info] = $env->step($action);
                $states = $this->customState($env,$states,$done,$truncated,$info);
                $reward = $this->customReward($env,$episodeSteps,$states,$reward,$done,$truncated,$info);
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

    protected function progressBar(
        string $title,
        int $iterNumber,
        int $numIterations,
        int $evalInterval,
        int $startTime,
        int $maxDot,
        ) : void
    {
        if($iterNumber<0) {
            $this->console("\r{$title} 1/{$numIterations} ");
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
        $this->console("\r{$title} {$iterNumber}/{$numIterations} [".
            str_repeat('.',$dot).str_repeat(' ',$maxDot-$dot).
            "] {$elapsed} sec. remaining:{$rem_string}  ");
    }
}
