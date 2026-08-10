<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;

class ActionMask
{
    public static function hasAny(object $la, NDArray $mask) : bool
    {
        $trueCount = $la->sum($mask);
        if ($trueCount instanceof NDArray) {
            $trueCount = $la->scalar($trueCount);
        }
        return (float)$trueCount>0.0;
    }
}
