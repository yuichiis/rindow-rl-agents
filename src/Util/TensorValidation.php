<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;

class TensorValidation
{
    public static function allFinite(object $la, NDArray $values) : bool
    {
        $finite = $la->isfinite($la->copy($values));
        return self::scalar($la,$la->min($finite))!=0.0;
    }

    public static function hasInf(object $la, NDArray $values) : bool
    {
        $infinite = $la->isinf($la->copy($values));
        return self::scalar($la,$la->max($infinite))!=0.0;
    }

    private static function scalar(object $la, mixed $value) : float
    {
        if ($value instanceof NDArray) {
            $value = $la->scalar($value);
        }
        return (float)$value;
    }
}
