<?php
namespace Rindow\RL\Agents\Env\Maze;

use Throwable;
use Interop\Polite\AI\RL\Environment;
use Interop\Polite\AI\RL\Spaces\Space;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

/** Transfers Maze dictionary observations and actions across the NN device boundary. */
class DeviceWrapper implements Environment
{
    public function __construct(
        private NeuralNetworks $nn,
        private Environment $env,
    ) {
    }

    public function maxEpisodeSteps() : int
    {
        return $this->env->maxEpisodeSteps();
    }

    public function rewardThreshold() : float
    {
        return $this->env->rewardThreshold();
    }

    public function observationSpace() : ?Space
    {
        return $this->env->observationSpace();
    }

    public function actionSpace() : ?Space
    {
        return $this->env->actionSpace();
    }

    /** @return array{array{location:NDArray,actionMask:NDArray},array<string,mixed>} */
    public function reset(?int $seed=null) : array
    {
        [$observation, $info] = $this->env->reset($seed);
        return [$this->deviceObservation($observation), $info];
    }

    /** @return array{array{location:NDArray,actionMask:NDArray},float,bool,bool,array<string,mixed>} */
    public function step(mixed $action) : array
    {
        if (!($action instanceof NDArray)) {
            $type = is_object($action) ? get_class($action) : gettype($action);
            throw new \InvalidArgumentException("Action must be NDArray. {$type} given.");
        }
        [$observation, $reward, $terminated, $truncated, $info] =
            $this->env->step($this->nn->hostArray($action));
        return [
            $this->deviceObservation($observation),
            $reward,
            $terminated,
            $truncated,
            $info,
        ];
    }

    /** @return array{location:NDArray,actionMask:NDArray} */
    private function deviceObservation(mixed $observation) : array
    {
        if (!is_array($observation)
            || !($observation['location'] ?? null) instanceof NDArray
            || !($observation['actionMask'] ?? null) instanceof NDArray) {
            throw new \UnexpectedValueException(
                'Maze observation must contain location and actionMask NDArrays.'
            );
        }
        return [
            'location'=>$this->nn->deviceArray($observation['location']),
            'actionMask'=>$this->nn->deviceArray($observation['actionMask']),
        ];
    }

    public function render(?string $mode=null) : mixed
    {
        return $this->env->render($mode);
    }

    public function show(?string $path=null, ?bool $loop=null, ?int $delay=null) : mixed
    {
        return $this->env->show($path, $loop, $delay);
    }

    public function close() : void
    {
        $this->env->close();
    }

    public function toString() : string
    {
        return $this->env->toString();
    }

    public function enter() : void
    {
        $this->env->enter();
    }

    public function exit(?Throwable $e=null) : bool
    {
        return $this->env->exit($e);
    }
}
