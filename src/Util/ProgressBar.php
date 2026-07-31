<?php
namespace Rindow\RL\Agents\Util;

class ProgressBar
{

    private ?string $lastConsoleOutput = null;
    private string $title;
    private int $numIterations;
    private int $maxDot;
    private int $startTime;

    /**
     * override this method to change the output destination of the progress bar.
     */
    protected function console(string $message)
    {
        if(defined('STDERR')) {
            fwrite(STDERR,$message);
        }
    }

    public function start(
        string $title,
        int $numIterations,
        int $maxDot,
    ) : void
    {
        $this->lastConsoleOutput = null;
        $this->title = $title;
        $this->numIterations = $numIterations;
        $this->maxDot = $maxDot;
        $this->startTime = time();
        $message = "\r{$this->title} 0/{$this->numIterations} [".str_repeat(' ',$this->maxDot)."] 0 sec. remaining:????  ";
        $this->console($message);
        $this->lastConsoleOutput = $message;
    }

    public function update(
        int $iterNumber,
    ) : void
    {
        if($iterNumber<1) {
            $message = "\r{$this->title} 0/{$this->numIterations} ";
            $this->console($message);
            $this->lastConsoleOutput = $message;
            return;
        }
        $elapsed = time() - $this->startTime;
        if($this->numIterations) {
            $completion = $iterNumber / $this->numIterations;
            $progressOfAgg = ((($iterNumber-1)%$this->numIterations)+1) / $this->numIterations;
            $estimated = $elapsed / $completion;
            $remaining = $estimated - $elapsed;
            $dot = (int)ceil($this->maxDot*$progressOfAgg);
            $sec = (int)floor($remaining) % 60;
            $min = (int)floor($remaining/60) % 60;
            $hour = (int)floor($remaining/3600);
            $rem_string = ($hour?$hour.':':'').sprintf('%02d:%02d',$min,$sec);
        } else {
            $dot = 1;
            $rem_string = '????';
            $this->console($this->maxDot."\n");
        }
        $message = "\r{$this->title} {$iterNumber}/{$this->numIterations} [".
            str_repeat('.',$dot).str_repeat(' ',$this->maxDot-$dot).
            "] {$elapsed} sec. remaining:{$rem_string}  ";
        $this->console($message);
        $this->lastConsoleOutput = $message;
    }

    public function clearProgressBar() : void
    {
        if($this->lastConsoleOutput===null) {
            return;
        }
        $message = "\r".str_repeat(' ',strlen($this->lastConsoleOutput)-1)."\r";
        $this->console($message);
    }

    public function retrieveProgressBar() : void
    {
        if($this->lastConsoleOutput===null) {
            return;
        }
        $this->console($this->lastConsoleOutput);
    }

    public function laptime() : int
    {
        return time() - $this->startTime;
    }

    public function laptimeString() : string
    {
        $elapsed = time() - $this->startTime;
        $sec = (int)floor($elapsed) % 60;
        $min = (int)floor($elapsed/60) % 60;
        $hour = (int)floor($elapsed/3600);
        return ($hour?$hour.':':'').sprintf('%02d:%02d',$min,$sec);
    }

}
