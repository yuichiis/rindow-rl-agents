<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Driver;
use Rindow\RL\Agents\EventManager;

class AnnealingEpsGreedy extends AbstractPolicy
{
    protected $qPolicy;
    protected $decayRate;
    protected $start;
    protected $stop;
    protected $currentTime = 0;
    protected $episodeAnnealing;

    public function __construct(
        object $la,
        float $start=null,float $stop=null,float $decayRate=null,
        bool $episodeAnnealing=null)
    {
        if($decayRate===null) {
            $decayRate = 0.01;
        }
        if($start===null) {
            $start = 0.9;
        }
        if($stop===null) {
            $stop = 0.1;
        }
        parent::__construct($la);
        $this->decayRate = $decayRate;
        $this->start = $start;
        $this->stop = $stop;
        $this->episodeAnnealing = $episodeAnnealing;
        $this->initialize();
    }

    public function setEpisodeAnnealing(bool $episodeAnnealing)
    {
        $this->episodeAnnealing = $episodeAnnealing;
    }

    public function register(EventManager $eventManager=null) : void
    {
        if($this->episodeAnnealing) {
            $eventManager->attach(Driver::EVENT_END_EPISODE,[$this,'updateTime']);
        }
    }

    public function initialize()
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
    * @param NDArray<any> $states
    * @return NDArray<int> $actions
    */
    public function action(QPolicy $qPolicy, NDArray $states, bool $training) : NDArray
    {
        $la = $this->la;
        if(!$training) {
            return $this->calcMaxValueActions($qPolicy, $states);
        }

        $epsilon = $this->getEpsilon();
        $threshold = (int)floor($epsilon * getrandmax());
        if($threshold > mt_rand()) {
            $actions = $qPolicy->sample($states);
        } else {
            $actions = $this->calcMaxValueActions($qPolicy, $states);
        }

        if(!$this->episodeAnnealing) {
            $this->currentTime++;
        }
        return $actions;
    }
}
