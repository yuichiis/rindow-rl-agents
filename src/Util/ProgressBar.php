<?php
namespace Rindow\RL\Agents\Util;

class ProgressBar
{
    private ?string $lastConsoleOutput = null;

    /**
     * override this method to change the output destination of the progress bar.
     */
    protected function console(string $message)
    {
        if(defined('STDERR')) {
            fwrite(STDERR,$message);
        }
    }

    public function progressBar(
        string $title,
        int $iterNumber,
        int $numIterations,
        int $startTime,
        int $maxDot,
        ) : void
    {
        if($iterNumber<1) {
            $message = "\r{$title} 0/{$numIterations} ";
            $this->console($message);
            $this->lastConsoleOutput = $message;
            return;
        }
        $elapsed = time() - $startTime;
        if($numIterations) {
            $completion = $iterNumber / $numIterations;
            $progressOfAgg = ((($iterNumber-1)%$numIterations)+1) / $numIterations;
            $estimated = $elapsed / $completion;
            $remaining = $estimated - $elapsed;
            $dot = (int)ceil($maxDot*$progressOfAgg);
            $sec = (int)floor($remaining) % 60;
            $min = (int)floor($remaining/60) % 60;
            $hour = (int)floor($remaining/3600);
            $rem_string = ($hour?$hour.':':'').sprintf('%02d:%02d',$min,$sec);
        } else {
            $dot = 1;
            $rem_string = '????';
            $this->console($maxDot."\n");
        }
        $message = "\r{$title} {$iterNumber}/{$numIterations} [".
            str_repeat('.',$dot).str_repeat(' ',$maxDot-$dot).
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

}
