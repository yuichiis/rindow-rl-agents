<?php
namespace Rindow\RL\Agents\Metrics;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\History as HistoryInterface;

class History implements HistoryInterface
{
    protected array $defaults = [
        'epsilon' => ['label'=>'eps','format'=>'%5.3f'],
        'steps' => ['label'=>'st','format'=>'%1.1f'],
        'reward' => ['label'=>'rw','format'=>'%1.1f'],
        'loss' => ['label'=>'loss','format'=>'%1.1f'],
        'valSteps' => ['label'=>'vSt','format'=>'%3.1f'],
        'valRewards' => ['label'=>'vRwd','format'=>'%3.1f'],
    ];
    protected array $metricValue = [];
    protected array $metricCount = [];
    protected array $history = [];
    protected array $attracted = [];

    public function attract(array $metrics) : void
    {
        if(array_is_list($metrics)) {
            $org = $metrics;
            $metrics = [];
            foreach($org as $name) {
                $metrics[$name] = ['name'=>$name];
                if(isset($this->defaults[$name])) {
                    $metrics[$name] = 
                        array_merge($metrics[$name],$this->defaults[$name]);
                } else {
                    $metrics[$name] = 
                        array_merge($metrics[$name],['label'=>$name,'format'=>'%3.3f']);
                }
            }
        }
        $this->attracted = $metrics;
    }

    public function isAttracted(string $name) : bool
    {
        return isset($this->attracted[$name]);
    }

    public function attracted() : array
    {
        return $this->attracted;
    }

    public function update(string $name, float $value) : void
    {
        if(!isset($this->metricCount[$name])) {
            $this->metricValue[$name] = 0.0;
            $this->metricCount[$name] = 0;
        }
        $this->metricValue[$name] += $value;
        $this->metricCount[$name] += 1;
    }

    public function result(string $name) : float
    {
        if(!isset($this->metricCount[$name]) ||
            $this->metricCount[$name]==0) {
            return 0.0;
        }
        $result = $this->metricValue[$name] / $this->metricCount[$name];
        return $result;
    }
    
    public function reset(string $name) : void
    {
        $this->metricValue[$name] = 0.0;
        $this->metricCount[$name] = 0;
    }

    public function add(string $name, float $value)
    {
        if(!isset($this->history[$name])) {
            $this->history[$name] = [];
        }
        $this->history[$name][] = $value;
    }

    public function resetAll() : void
    {
        $this->metricValue = [];
        $this->metricCount = [];
    }

    public function record() : void
    {
        foreach($this->attracted as $name => $attr) {
            $this->add($name,$this->result($name));
        }
    }

    public function history() : array
    {
        return $this->history;
    }

    public function render(?array $exclude=null) : string
    {
        $exclude ??= [];
        $string = '';
        foreach($this->attracted as $name => $attr) {
            if(in_array($name,$exclude)) {
                continue;
            }
            if($string!=='') {
                $string .= ' ';
            }
            $label = $attr['label'];
            $value = isset($attr['format']) ? sprintf($attr['format'],$this->result($name)) : $this->result($name);
            $string .= "$label=$value";
        }
        return $string;
    }
}