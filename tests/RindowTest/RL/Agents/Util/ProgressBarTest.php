<?php
namespace RindowTest\RL\Agents\Util\ProgressBarTest;

use PHPUnit\Framework\TestCase;
use Rindow\RL\Agents\Util\ProgressBar;

class ProgressBarTest extends TestCase
{
    public function testProgressBar()
    {
        $progressBar = new ProgressBar();

        $progressBar->progressBar('Test', 0, 10, time(), 10);
        $progressBar->progressBar('Test', 1, 10, time(), 10);
        $progressBar->progressBar('Test', 5, 10, time(), 10);
        $progressBar->progressBar('Test', 10, 10, time(), 10);
        
        $progressBar->clearProgressBar();
        $progressBar->retrieveProgressBar();
        $this->assertTrue(true);
    }
}
