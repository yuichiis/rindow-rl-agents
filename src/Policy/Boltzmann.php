<?php
namespace Rindow\RL\Agents\Policy;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\RL\Agents\Estimator;

class Boltzmann extends AbstractPolicy
{
    protected float $tau;
    protected float $min;
    protected float $max;
    protected bool $fromLogits;

    public function __construct(
        object $la,
        ?float $tau=null,
        ?float $min=null,
        ?float $max=null,
        ?bool $fromLogits=null,
        )
    {
        $fromLogits ??= true;
        $tau ??= 1.0;
        if($fromLogits) {
            $min ??= -500.0;
        } else {
            $min ??= 0;
            if($min<0) {
                throw new InvalidArgumentException("min must be greater than zero or zero without fromlogits.: $min given.");
            }
        }
        $max ??= 500.0;

        parent::__construct($la);
        $this->tau = $tau;
        $this->min = $min;
        $this->max = $max;
        $this->fromLogits = $fromLogits;
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    public function initialize() : void
    {
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
        //echo "boltz\n";

        // get values
        $actionPolicies = $estimator->getActionValues($states);
        //var_dump($actionPolicies->toArray());
        $actionPolicies = $la->copy($actionPolicies);
        if($this->tau!=1.0) {
            $actionPolicies = $la->pow($actionPolicies,$this->tau);
        }
        //echo "tau:".implode(',',$actionValues->toArray()[0])."\n";
        //echo "Policies:".$la->toString($actionPolicies,format:'%.3f')."\n";

        $actionPolicies = $la->minimum($la->maximum($actionPolicies,$this->min),$this->max);

        if(!$this->fromLogits) {
            $actionPolicies = $la->log($actionPolicies);
        }

        if($masks) {
            //echo "masking\n";
            $la->masking($masks,$actionPolicies,fill:-INF);
            //var_dump($masks->toArray());
            //var_dump($actionPolicies->toArray());
        }

        // random choice with logits
        $actions = $this->randomCategorical($actionPolicies);
        return $actions;
    }
}
