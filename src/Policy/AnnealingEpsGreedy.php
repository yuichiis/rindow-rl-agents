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
    protected $numActions;
    protected $currentTime = 0;
    protected $episodeAnnealing;

    public function __construct(
        object $la, QPolicy $qPolicy,
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
        $this->qPolicy = $qPolicy;
        $this->decayRate = $decayRate;
        $this->start = $start;
        $this->stop = $stop;
        $this->numActions = $this->qPolicy->numActions();
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
    * @param Any $states
    * @return Any $action
    */
    public function action($state,bool $training)
    {
        $epsilon = $this->getEpsilon();
        $threshold = (int)floor($epsilon * getrandmax());
        $numActions = $this->numActions;
        if($training && $threshold > mt_rand()) {
            $action = $this->qPolicy->sample($state);
        } else {
            $qValues = $this->qPolicy->getQValues($state);
            $action = $this->la->imax($qValues);
        }
        if($training) {
            if(!$this->episodeAnnealing) {
                $this->currentTime++;
            }
        }
        return $action;
    }
}
