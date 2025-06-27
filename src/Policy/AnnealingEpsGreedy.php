<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Driver;
use Rindow\RL\Agents\EventManager;

class AnnealingEpsGreedy extends AbstractPolicy
{
    protected float $start;
    protected float $stop;
    protected float $decayRate;
    protected bool $episodeAnnealing;
    protected int $currentTime = 0;

    public function __construct(
        object $la,
        ?float $start=null,
        ?float $stop=null,
        ?float $decayRate=null,
        ?bool $episodeAnnealing=null,
        )
    {
        $start ??= 0.9;
        $stop ??= 0.1;
        $decayRate ??= 0.01;
        $episodeAnnealing ??= false;

        parent::__construct($la);
        $this->start = $start;
        $this->stop = $stop;
        $this->decayRate = $decayRate;
        $this->episodeAnnealing = $episodeAnnealing;
        $this->initialize();
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    public function setEpisodeAnnealing(bool $episodeAnnealing)
    {
        $this->episodeAnnealing = $episodeAnnealing;
    }

    public function register(?EventManager $eventManager=null) : void
    {
        if($this->episodeAnnealing) {
            $eventManager->attach(Driver::EVENT_END_EPISODE,[$this,'updateTime']);
        }
    }

    public function initialize() : void
    {
        $this->currentTime = 0;
    }

    public function updateTime(array $args)
    {
        $this->currentTime++;
    }

    public function getEpsilon()
    {
        $time = $this->currentTime;
        return $this->stop + ($this->start-$this->stop)*exp(-$this->decayRate*$time);
    }

    /**
    * param  NDArray $states  : (batches,...StateDims) typeof int32 or float32
    * return NDArray $actions : (batches) typeof int32
    */
    public function actions(Estimator $estimator, NDArray $states, bool $training, ?NDArray $masks) : NDArray
    {
        $la = $this->la;
        if(!$training) {
            return $this->calcMaxValueActions($estimator, $states, $masks);
        }

        $epsilon = $this->getEpsilon();
        $threshold = (int)floor($epsilon * getrandmax());
        $rand = mt_rand();
        if($threshold > $rand) {
            $actions = $this->calcRandomActions($estimator, $states, $masks);
        } else {
            $actions = $this->calcMaxValueActions($estimator, $states, $masks);
        }

        if(!$this->episodeAnnealing) {
            $this->currentTime++;
        }
        return $actions;
    }
}
