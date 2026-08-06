<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Agents\Agent\QLearning\QLearningAgent;
use Rindow\RL\Agents\Agent\QLearning\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Gym\ClassicControl\MountainCar\MountainCarV0;

const TOTAL_EPISODES=2000, EVAL_EVERY=25, EVAL_EPISODES=10;
const MODEL_FILE=__DIR__.'/../models/mountaincar-qlearning.weights';
$mo=new MatrixOperator(); $la=$mo->laRawMode();
$seedText=getenv('RL_SEED'); $seed=$seedText===false?null:(int)$seedText;
if($seed!==null) $la->setSeed($seed);
echo 'Random seed: '.($seed??'system default')."\n";
$env=new MountainCarV0($la); $evalEnv=new MountainCarV0($la);
if($seed!==null) { $env->observationSpace()->seed($seed); $env->actionSpace()->seed($seed);
    $evalEnv->observationSpace()->seed($seed+1); $evalEnv->actionSpace()->seed($seed+1); }
$coder=new TileCoder([-1.2,-0.07],[0.6,0.07],8,8);
$agent=new QLearningAgent($la,$coder,$env->actionSpace()->n(),0.3,1.0,0.0);
$runner=new Runner($la,$env,$evalEnv,$agent,-110.0);
$modelFile=getenv('RL_MODEL_FILE')?:MODEL_FILE;
if(is_file($modelFile)) { $agent->loadWeightsFromFile($modelFile); echo "Model loaded: {$modelFile}\n"; }
else {
    $runner->train((int)(getenv('RL_TOTAL_EPISODES')?:TOTAL_EPISODES),
        (int)(getenv('RL_EVAL_EVERY')?:EVAL_EVERY),EVAL_EPISODES,$modelFile);
    if(is_file($modelFile)) $agent->loadWeightsFromFile($modelFile);
}
if(getenv('RL_SKIP_DEMO')!=='1') {
    for($episode=1;$episode<=5;$episode++) { [$obs]=$env->reset(); $done=false; $total=0; $steps=0; $env->render();
        while(!$done) { $action=$la->array($agent->selectActionDeterministic($obs),dtype:NDArray::int32);
            [$obs,$reward,$terminated,$truncated]=$env->step($action); $done=$terminated||$truncated;
            $total+=$reward; $steps++; $env->render(); }
        printf("Test Episode %d | Steps=%d | RawReward=%+.1f\n",$episode,$steps,$total); }
    echo 'filename: '.$env->show(path:__DIR__.'/../graphics/mountaincar-qlearning-trained.gif')."\n";
}
