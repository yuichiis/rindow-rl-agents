<?php
require __DIR__.'/../vendor/autoload.php';
require __DIR__.'/include/env.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Agent\QLearning\QLearningAgent;
use Rindow\RL\Agents\Agent\QLearning\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Agents\Env\MountainCar\DeviceWrapper;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const SEED = 42;
const TOTAL_EPISODES=2000, EVAL_EVERY=25, EVAL_EPISODES=10;
const MODEL_FILE=__DIR__.'/../models/mountaincar-qlearning.weights';
$seed = rlEnvInt('RL_SEED',SEED);
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $nn->la();
$hostLa = $mo->laRawMode();
$la->setSeed($seed);
echo "Random seed: {$seed}\n";
echo 'Accelerated: '.($la->accelerated() ? 'true' : 'false')."\n";

$env=new MountainCarV0($hostLa); $evalEnv=new MountainCarV0($hostLa);
rlSeedSpaces($env,$evalEnv,$seed);
if($la->accelerated()) {
    $env=new DeviceWrapper($nn,$env); $evalEnv=new DeviceWrapper($nn,$evalEnv);
}
$coder=new TileCoder([-1.2,-0.07],[0.6,0.07],8,8);
$agent=new QLearningAgent($la,$coder,$env->actionSpace()->n(),0.3,1.0,0.0,nn:$nn);
$runner=new Runner($la,$env,$evalEnv,$agent,-110.0);
$modelFile=rlEnvString('RL_MODEL_FILE',MODEL_FILE);
$evalEpisodes = rlEnvInt('RL_EVAL_EPISODES',EVAL_EPISODES);
if(is_file($modelFile)) { $agent->loadWeightsFromFile($modelFile); echo "Model loaded: {$modelFile}\n"; }
else {
    $runner->train(rlEnvInt('RL_TOTAL_EPISODES',TOTAL_EPISODES),
        rlEnvInt('RL_EVAL_EVERY',EVAL_EVERY),$evalEpisodes,$modelFile);
    if(is_file($modelFile)) $agent->loadWeightsFromFile($modelFile);
}
if(!rlEnvBool('RL_SKIP_DEMO')) {
    for($episode=1;$episode<=5;$episode++) { [$obs]=$env->reset(); $done=false; $total=0; $steps=0; $env->render();
        while(!$done) { $action=$la->array($agent->selectActionDeterministic($obs),dtype:NDArray::int32);
            [$obs,$reward,$terminated,$truncated]=$env->step($action); $done=$terminated||$truncated;
            $total+=$reward; $steps++; $env->render(); }
        printf("Test Episode %d | Steps=%d | RawReward=%+.1f\n",$episode,$steps,$total); }
    echo 'filename: '.$env->show(path:__DIR__.'/../graphics/mountaincar-qlearning-trained.gif')."\n";
}
