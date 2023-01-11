<?php
namespace Rindow\RL\Agents\Network;

use Rindow\RL\Agents\QPolicy;
use Rindow\RL\Agents\Util\Random;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class QTable implements QPolicy
{
    use Random;

    const MODEL_FILENAME = '%s.model';

    protected $rules;
    protected $obsSize;
    protected $numActions;
    protected $rulesProbs;
    protected $table;

    public function __construct($la, NDArray $rules)
    {
        $this->la = $la;
        $this->rules = $rules;
        [$obsSize,$numActions] = $rules->shape();
        $this->obsSize = $obsSize;
        $this->numActions = $numActions;
        // initialize legal actions probabilities
        $this->rulesProbs = $this->generateProbabilities($this->rules);
        $this->initialize();
    }

    public function initialize() // : Operation
    {
        $la = $this->la;
        // initialize Q table
        $this->table = $la->randomUniform([$this->obsSize,$this->numActions],0,1);
        if($this->rules) {
            $la->multiply($this->rules,$this->table);
        }
    }

    public function obsSize()
    {
        return $this->obsSize;
    }

    public function numActions() : int
    {
        return $this->numActions;
    }

    public function table()
    {
        return $this->table;
    }

    //public function getQValues($state) : NDArray
    //{
    //    if(!is_int($state)) {
    //        throw new InvalidArgumentException('state must be integer');
    //    }
    //    return $this->table[$state];
    //}

    public function getQValues(NDArray $states) : NDArray
    {
        $la = $this->la;
        if($states->ndim()!=2) {
            throw new InvalidArgumentException('states must be 2D integer');
        }
        //return $this->table[$state];
        [$count,$size] = $states->shape();
        if($size!=1) {
            throw new InvalidArgumentException('size of state detail must be 1');
        }
        //$values = $la->alloc([$count,$size]);
        //foreach ($states as $key => $state) {
        //    $i = $state[0];
        //    $la->copy($this->table[[$i,$i]],$values[$key]);
        //}
        $states = $la->squeeze($states,$axis=-1);
        $values = $la->gather($this->table,$states,$axis=null);
        return $values;
    }

    public function sample(NDArray $states) : NDArray
    {
        $la = $this->la;
        $states = $la->squeeze($states,$axis=-1);
        $probs = $la->gather($this->rulesProbs,$states,$axis=null);
        $actions = $this->randomCategorical($probs,1);
        return $actions;
    }

    public function fileExists(string $filename) : bool
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        return file_exists($filename);
    }

    public function setPortableSerializeMode(bool $mode) : void
    {
        $this->table->setPortableSerializeMode($mode);
    }

    public function saveWeightsToFile($filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $dump = serialize($this->table);
        file_put_contents($filename, $dump);
    }

    public function loadWeightsFromFile($filename) : void
    {
        $filename = sprintf(self::MODEL_FILENAME,$filename);
        $dump = file_get_contents($filename);
        $q = unserialize($dump);
        $this->table = $q;
    }
}
