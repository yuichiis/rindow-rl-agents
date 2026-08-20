<?php
namespace RindowTest\RL\Agents\Util\ProgressBarTest;

use PHPUnit\Framework\TestCase;
use Rindow\RL\Agents\Util\ProgressBar;

class ProgressBarTest extends TestCase
{
    public function testProgressBar()
    {
        $progressBar = new ProgressBar();

        $progressBar->start('Test', 10, 10);
        $progressBar->update(0);
        $progressBar->update(1);
        $progressBar->update(5);
        $progressBar->update(10);

        $progressBar->clearProgressBar();
        $progressBar->retrieveProgressBar();
        $this->assertTrue(true);
    }
}
