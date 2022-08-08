<?php
namespace Rindow\RL\Agents\ReplayBuffer;

use SplFixedArray;
use InvalidArgumentException;
use LogicException;
use Rindow\RL\Agents\ReplayBuffer as ReplayBufferInterface;

class ReplayBuffer implements ReplayBufferInterface
{
    protected $la;
    protected $maxSize;
    protected $size = 0;
    protected $last = -1;

    public function __construct($la,$maxSize)
    {
        $this->la = $la;
        if($maxSize<=0) {
            throw new InvalidArgumentException('maxSize must be greater then 0');
        }
        $this->maxSize = $maxSize;
        $this->array = new SplFixedArray($maxSize);
    }

    public function clear()
    {
        $this->array = new SplFixedArray($this->maxSize);
        $this->size = 0;
        $this->last = -1;
    }

    public function size() : int
    {
        return $this->size;
    }

    public function count() : int
    {
        return $this->size();
    }

    public function add($item) : void
    {
        $this->last++;
        if($this->last >= $this->maxSize) {
            $this->last=0;
        }
        if($this->last >= $this->size) {
            $this->size = $this->last+1;
        }
        $this->array[$this->last] = $item;
    }

    public function last()
    {
        if($this->last<0) {
            throw new LogicException('No data');
        }
        return $this->array[$this->last];
    }

    public function recently(int $quantity) : iterable
    {
        if($this->last<0) {
            throw new LogicException('No data');
        }
        if($quantity<=0) {
            throw new InvalidArgumentException('quantity must be greater then 0');
        }
        if($quantity > $this->size) {
            throw new InvalidArgumentException('Too few items are stored.');
        }
        $pos=$this->last-$quantity+1;
        if($pos<0) {
            $pos += $this->size;
        }
        $recently = [];
        for($i=0;$i<$quantity;$i++,$pos++) {
            if($pos>=$this->size) {
                $pos = 0;
            }
            $recently[] = $this->array[$pos];
        }
        return $recently;
    }

    public function sample(int $quantity) : iterable
    {
        if($quantity<=0) {
            throw new InvalidArgumentException('quantity must be greater then 0');
        }
        if($quantity > $this->size) {
            throw new InvalidArgumentException('Too few items are stored.');
        }
        $indexes = $this->la->randomSequence($this->size,$quantity);
        $samples = [];
        foreach ($indexes as $idx) {
            $samples[] = $this->array[$idx];
        }
        return $samples;
    }
}
