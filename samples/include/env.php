<?php

function rlEnvInt(string $name, int $default) : int
{
    $value = getenv($name);
    return $value === false ? $default : (int)$value;
}

function rlEnvFloat(string $name, float $default) : float
{
    $value = getenv($name);
    return $value === false ? $default : (float)$value;
}

function rlEnvString(string $name, string $default) : string
{
    $value = getenv($name);
    return $value === false ? $default : $value;
}

function rlEnvBool(string $name, bool $default=false) : bool
{
    $value = getenv($name);
    if ($value === false) {
        return $default;
    }
    return filter_var($value,FILTER_VALIDATE_BOOL,FILTER_NULL_ON_FAILURE) ?? $default;
}

function rlSeedSpaces(object $env, object $evalEnv, int $seed) : void
{
    $env->observationSpace()->seed($seed);
    $env->actionSpace()->seed($seed);
    $evalEnv->observationSpace()->seed($seed+1);
    $evalEnv->actionSpace()->seed($seed+1);
}
