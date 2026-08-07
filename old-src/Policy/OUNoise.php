<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\Runner;
use Rindow\RL\Agents\Util\OUProcess;
use Rindow\RL\Agents\EventManager;
use function Rindow\Math\Matrix\R;

class OUNoise extends AbstractPolicy
{
    protected Estimator $estimator;
    protected NDArray $lower_bound;
    protected NDArray $upper_bound;
    protected OUProcess $noise;
    protected bool $episodeAnnealing;
    protected NDArray $std_dev;
    protected ?float $noize_decay;
    protected ?NDArray $min_std_dev;

    public function __construct(
        object $la,
        NDArray $mean,
        NDArray $std_dev,
        NDArray $lower_bound,
        NDArray $upper_bound,
        ?float $theta=null,
        ?float $dt=null,
        ?NDArray $x_initial=null,
        ?float $noise_decay=null,
        ?NDArray $min_std_dev=null,
        ?bool $episodeAnnealing=null,
        )
    {
        $episodeAnnealing ??= false;

        parent::__construct($la);
        $this->std_dev = $la->copy($std_dev);
        $this->lower_bound = $lower_bound;
        $this->upper_bound = $upper_bound;
        $this->noize_decay = $noise_decay;

        if($min_std_dev!==null) {
            if($std_dev->shape()!=$min_std_dev->shape()) {
                throw new InvalidArgumentException("shape of std_dev and min_std_dev must be the same");
            }
        }
        $this->min_std_dev = $min_std_dev;
        $this->noise = new OUProcess($la,
                $mean,$this->std_dev,
                $theta,$dt,
                $x_initial);
        $this->episodeAnnealing = $episodeAnnealing;

    }

    public function isContinuousActions() : bool
    {
        return true;
    }

    public function setEpisodeAnnealing(bool $episodeAnnealing)
    {
        $this->episodeAnnealing = $episodeAnnealing;
    }

    public function register(?EventManager $eventManager=null) : void
    {
        if($this->episodeAnnealing) {
            $eventManager->attach(Runner::EVENT_END_EPISODE,[$this,'updateTime']);
        }
    }

    public function initialize() : void
    {
        $this->noise->reset();
    }

    public function updateTime(array $args)
    {
        if($this->noize_decay===null) {
            return;
        }
        $la = $this->la;
        $la->scal($this->noize_decay,$this->std_dev);
        $la->maximum($this->std_dev,$this->min_std_dev);
    }

    public function getEpsilon()
    {
        $la = $this->la;
        $eps = $la->scalar($la->squeeze($this->std_dev[[0,1]]));
        return $eps;
    }

    /**
    * param  NDArray $states  : (batches,...StateDims)  typeof int32 or float32
    * return NDArray $actions : (batches,...ActionDims) typeof float32
    */
    public function actions(Estimator $estimator, NDArray $states, bool $training, ?NDArray $masks) : NDArray
    {
        $la = $this->la;
        $actions = $estimator->getActionValues($states);

        if($training) {
            $noise = $this->noise->process();
            $actions = $la->add($noise,$la->copy($actions)); // add noise to batch
        }

        $orgShape = $shape = $actions->shape();
        $count = array_shift($shape);
        $size = array_product($shape);
        $actions = $la->transpose($actions->reshape([$count,$size]));

        $flat_lower = $this->lower_bound->reshape([$size]);
        $flat_upper = $this->upper_bound->reshape([$size]);
        for($i=0;$i<$size;$i++) {
            $la->maximum($actions[R($i,$i+1)],$flat_lower[$i]);
            $la->minimum($actions[R($i,$i+1)],$flat_upper[$i]);
        }

        $actions = $la->transpose($actions);
        $actions = $actions->reshape($orgShape);

        if(!$this->episodeAnnealing) {
            $this->updateTime([]);
        }
        return $actions;
    }
}
