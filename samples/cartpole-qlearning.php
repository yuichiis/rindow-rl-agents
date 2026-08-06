<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Agents\Agent\QLearning\QLearningAgent;
use Rindow\RL\Agents\Agent\QLearning\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV1;

const TOTAL_EPISODES=2000, EVAL_EVERY=25, EVAL_EPISODES=10;
const MODEL_FILE=__DIR__.'/../models/cartpole-qlearning.weights';
$mo=new MatrixOperator(); $la=$mo->laRawMode(); $seed=(int)(getenv('RL_SEED')?:42); $la->setSeed($seed);
echo "Random seed: {$seed}\n";
$env=new CartPoleV1($la); $evalEnv=new CartPoleV1($la);
$env->observationSpace()->seed($seed); $env->actionSpace()->seed($seed);
$evalEnv->observationSpace()->seed($seed+1); $evalEnv->actionSpace()->seed($seed+1);
$coder=new TileCoder([-2.4,-3.0,-0.2095,-3.5],[2.4,3.0,0.2095,3.5],8,8);
$agent=new QLearningAgent($la,$coder,$env->actionSpace()->n(),0.1,0.99,0.05,initialValue:100.0);
$runner=new Runner($la,$env,$evalEnv,$agent,475.0);
$modelFile=getenv('RL_MODEL_FILE')?:MODEL_FILE;
if(is_file($modelFile)) { $agent->loadWeightsFromFile($modelFile); echo "Model loaded: {$modelFile}\n"; }
else { $runner->train((int)(getenv('RL_TOTAL_EPISODES')?:TOTAL_EPISODES),
    (int)(getenv('RL_EVAL_EVERY')?:EVAL_EVERY),EVAL_EPISODES,$modelFile);
    if(is_file($modelFile)) $agent->loadWeightsFromFile($modelFile); }
if(getenv('RL_SKIP_DEMO')!=='1') {
    for($episode=1;$episode<=5;$episode++) { [$obs]=$env->reset(); $done=false; $total=0; $steps=0; $env->render();
        while(!$done) { $action=$la->array($agent->selectActionDeterministic($obs),dtype:NDArray::int32);
            [$obs,$reward,$terminated,$truncated]=$env->step($action); $done=$terminated||$truncated;
            $total+=$reward; $steps++; $env->render(); }
        printf("Test Episode %d | Steps=%d | RawReward=%+.1f\n",$episode,$steps,$total); }
    echo 'filename: '.$env->show(path:__DIR__.'/../graphics/cartpole-qlearning-trained.gif')."\n"; }
