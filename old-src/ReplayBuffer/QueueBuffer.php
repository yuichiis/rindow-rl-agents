<?php
namespace Rindow\RL\Agents\ReplayBuffer;

use InvalidArgumentException;
use LogicException;
use Rindow\RL\Agents\ReplayBuffer as ReplayBufferInterface;

class QueueBuffer implements ReplayBufferInterface
{
    protected object $la;
    protected int $maxSize;
    protected int $size = 0;
    protected ?array $queueSet=null;

    public function __construct(object $la, int $maxSize)
    {
        $this->la = $la;
        if($maxSize<=0) {
            throw new InvalidArgumentException('maxSize must be greater then 0');
        }
        $this->maxSize = $maxSize;
    }

    public function clear() : void
    {
        if($this->queueSet===null) {
            return;
        }
        foreach($this->queueSet as $key => $queue) {
            $this->queueSet[$key] = [];
        }
        $this->size = 0;
    }

    public function size() : int
    {
        return $this->size;
    }

    public function maxSize() : int
    {
        return $this->maxSize;
    }

    public function count() : int
    {
        return $this->size;
    }

    public function add(array $items) : void
    {
        if($this->queueSet===null) {
            $this->queueSet = [];
            foreach($items as $key => $value) {
                $this->queueSet[$key] = [];
            }
        } else {
            if(count($this->queueSet)!=count($items)) {
                throw new InvalidArgumentException('The number of fields must be the same.');
            }
        }
        foreach($items as $key => $value) {
            if(array_key_exists($key,$this->queueSet)==false) {
                throw new InvalidArgumentException('The field $idx is not defined.');
            }
            $this->queueSet[$key][] = $value;
        }
        $this->size++;
        if($this->size <= $this->maxSize) {
            return;
        }
        foreach($this->queueSet as $key => $queue) {
            array_shift($this->queueSet[$key]);
        }
        $this->size--;
    }

    public function get(int $index) : array
    {
        if($this->queueSet===null) {
            throw new LogicException('No data');
        }
        if($index < 0) {
            $index = $this->size + $index;
        }
        if($index < 0 || $index >= $this->size) {
            throw new InvalidArgumentException("$index is out of range.");
        }
        $set = [];
        foreach($this->queueSet as $key => $queue) {
            $set[$key] = $queue[$index];
        }
        return $set;
    }

    public function last() : array
    {
        return $this->get(-1);
    }

    public function recently(int $quantity) : iterable
    {
        if($quantity<=0) {
            throw new InvalidArgumentException('quantity must be greater then 0');
        }
        if($quantity > $this->size) {
            throw new InvalidArgumentException('Too few items are stored.');
        }
        $recently = [];
        foreach($this->queueSet as $key => $queue) {
            $recently[$key] = array_slice($queue,-$quantity);
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
        foreach($this->queueSet as $key => $queue) {
            $sample = [];
            foreach($indexes as $idx) {
                $sample[] = $queue[$idx];
            }
            $samples[$key] = $sample;
        }
        return $samples;
    }
}
