<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Agents\Agent\QLearning\QLearningAgent;
use Rindow\RL\Agents\Agent\QLearning\Runner;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;

const WIDTH=3, HEIGHT=3, EXIT_STATE=8, MAX_EPISODE_STEPS=100;
const TOTAL_EPISODES=500, EVAL_EVERY=10, EVAL_EPISODES=10;
const MODEL_FILE=__DIR__.'/../models/maze-qlearning.weights';
$mo=new MatrixOperator(); $la=$mo->laRawMode(); $seed=(int)(getenv('RL_SEED')?:1234); $la->setSeed($seed);
echo "Random seed: {$seed}\n";
$env=new Maze($la,null,WIDTH,HEIGHT,EXIT_STATE,true,MAX_EPISODE_STEPS);
$evalEnv=new Maze($la,$env->mazeRules(),WIDTH,HEIGHT,EXIT_STATE,true,MAX_EPISODE_STEPS);
$env->actionSpace()->seed($seed); $env->observationSpace()->seed($seed);
$evalEnv->actionSpace()->seed($seed+1); $evalEnv->observationSpace()->seed($seed+1);
$coder=new TileCoder([0.0,0.0],[HEIGHT-1.0,WIDTH-1.0],4,2);
$agent=new QLearningAgent($la,$coder,$env->actionSpace()->n(),0.2,1.0,0.1,
    stateField:'location',actionMaskField:'actionMask');
$runner=new Runner($la,$env,$evalEnv,$agent);
$modelFile=getenv('RL_MODEL_FILE')?:MODEL_FILE;
if(is_file($modelFile)) { $agent->loadWeightsFromFile($modelFile); echo "Model loaded: {$modelFile}\n"; }
else { $runner->train((int)(getenv('RL_TOTAL_EPISODES')?:TOTAL_EPISODES),
    (int)(getenv('RL_EVAL_EVERY')?:EVAL_EVERY),EVAL_EPISODES,$modelFile);
    if(is_file($modelFile)) $agent->loadWeightsFromFile($modelFile); }
if(getenv('RL_SKIP_DEMO')!=='1') { [$obs]=$env->reset(); $done=false; $total=0; $steps=0; $env->render();
    while(!$done) { $action=$la->array($agent->selectActionDeterministic($obs),dtype:NDArray::int32);
        [$obs,$reward,$terminated,$truncated]=$env->step($action); $done=$terminated||$truncated;
        $total+=$reward; $steps++; $env->render(); }
    printf("Test Episode 1 | Steps=%d | RawReward=%+.1f\n",$steps,$total);
    echo 'filename: '.$env->show(path:__DIR__.'/../graphics/maze-qlearning-trained.gif',delay:100)."\n"; }
