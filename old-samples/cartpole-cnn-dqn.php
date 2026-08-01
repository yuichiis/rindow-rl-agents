<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\CartPole\CartPoleV0;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Runner\StepRunner;
use Rindow\RL\Agents\Agent\DQN\DQN;
use Rindow\RL\Agents\Agent\DQN\QNetwork;
//use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\Util\GDUtil;

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

function get_cart_location($screen_width,$state)
{
    $x_threshold = 2.4;
    $world_width = $x_threshold * 2;
    $scale = $screen_width / $world_width;
    return (int)floor($state[0] * $scale + $screen_width / 2.0);
}

function get_screen($la,$state,$env)
{
    //$screen = $env->render($mode='rgb_array');
    $gd = $env->render($mode='handler');
    # Cart is in the lower half, so strip off the top and bottom of the screen
    //[$screen_height, $screen_width, $ch] = $screen->shape();
    [$screen_height, $screen_width, $ch] = [imagesy($gd),imagesx($gd),3];

    $top = (int)floor($screen_height*0.4);
    $bottom = (int)floor($screen_height * 0.8);
    $view_width = (int)floor($screen_width * 0.6);
    $cart_location = get_cart_location($screen_width,$state);
    if($cart_location < (int)floor($view_width/2)) {
        $slice_range = [0,$view_width];
    } elseif ($cart_location > ($screen_width - (int)floor($view_width/2))) {
        $slice_range = [$screen_width-$view_width, $screen_width];
    } else {
        $slice_range = [$cart_location - (int)floor($view_width / 2),
                            $cart_location + (int)floor($view_width / 2)];
    }
    # Strip off the edges, so that we have a square image centered on a cart
    [$begin,$end] = $slice_range;

    //$screen = $screen[R($top,$bottom+1)];
    //$screen = $la->slice($screen,[0,$begin],[$bottom-$top,$end-$begin]);

    $new_gd = imagecreatetruecolor(90,40);
    imagecopyresampled(
    //imagecopyresized(
        $new_gd,$gd,
        0,0,    $begin,$top,
        90,40,  $end-$begin,$bottom-$top
    );
    $util = new GDUtil($la);
    $screen = $util->getArray($new_gd);
    imagedestroy($new_gd);

    return $screen;
}

$last_screen_container = new stdClass();
$last_screen_container->last_screen = [];
$customStateFunction = function($env,$state,$done) use ($la,$last_screen_container) {
    $current_screen = get_screen($la,$state,$env);
    $current_screen = $la->astype($current_screen,NDArray::float32);
    if(isset($last_screen_container->last_screen[spl_object_id($env)])) {
        $state = $la->axpy($last_screen_container->last_screen[spl_object_id($env)],
            $la->copy($current_screen),-1);
    } else {
        $state = $la->zeros($la->alloc($current_screen->shape(),$current_screen->dtype()));
    }
    if($done) {
        $last_screen = null;
    } else {
        $last_screen = $current_screen;
    }
    $last_screen_container->last_screen[spl_object_id($env)] = $last_screen;
    return $state;
};

//$stateShape = [160, 360, 3];
//$numActions = 2;
## { $numIterations = 300; $evalInterval = 50; $numEvalEpisodes = 10;
##   $maxExperienceSize = 10000; $batchSize = 32;
##   $gamma = 0.99; $fcLayers = [100]; $targetUpdatePeriod = 5; $targetUpdateTau = 0.025;
##   $learningRate = 1e-3; $epsStart = 1.0; $epsStop = 0.05; $decayRate = 0.001;
##   $ddqn = false; $lossFn = null;}
## { $numIterations = 300; $evalInterval = 10; $numEvalEpisodes = 10;
##   $maxExperienceSize = 10000; $batchSize = 64;
##   $gamma = 0.99; $fcLayers = [100]; $targetUpdatePeriod = 5; $targetUpdateTau = 1.0;
##   $learningRate = 1e-3; $epsStart = 1.0; $epsStop = 0.05; $decayRate = 0.001;
##   $ddqn = true; $lossFn = $nn->losses->MeanSquaredError();}

$numIterations = 2000;#50;#20000;#300;  # 1000;#
$logInterval =   1;  #1; # 200;  #10; #
$evalInterval =  100;#5; # 1000; #10; #
$numEvalEpisodes = 10;
$maxExperienceSize = 10000;#100000;
$batchSize = 128;#32;#64;#
$gamma = 0.999;#1.0;#0.99;#
//$convLayers = [
//    [16,5,['strides'=>2,'batch_norm'=>[],'activation'=>'relu','pooling'=>[]]],
//    [32,5,['strides'=>2,'batch_norm'=>[],'activation'=>'relu','pooling'=>[]]],
//    [64,1,['strides'=>2,'batch_norm'=>[],'activation'=>'relu','pooling'=>[]]],
//    [256,1,['strides'=>2,'batch_norm'=>[],'activation'=>'relu','global_pooling'=>[]]],
//];
$convLayers = [
    [16,5,['strides'=>2,'batch_norm'=>[],'activation'=>'relu']],
    [32,5,['strides'=>2,'batch_norm'=>[],'activation'=>'relu']],
    [32,5,['strides'=>2,'batch_norm'=>[],'activation'=>'relu']],
];
$convType = '2d';
$fcLayers = [];# [256];# [32,32];#
$targetUpdatePeriod = 10; #5;  #5;    #5;   #5;   # 200;#
$targetUpdateTau =    1.0;#1.0;#0.025;#0.01;#0.05;#1.0;#
$learningRate = 1e-3;#1e-5;#
$epsStart = 1.0; #1.0; #0.9;#1.0; #
$epsStop =  0.05;#0.01;#0.1;#0.05;#
$decayRate = 0.01;#0.001;#0.0005;#
$ddqn = false;#true;#
$lossFn = null;#$nn->losses->MeanSquaredError();
//$optimizer = $nn->optimizers->RMSprop(lr:$learningRate);

$env = new CartPoleV0($la);
[$state,$info] = $env->reset();
$image = get_screen($la,$state,$env);
$stateShape = $image->shape();
$numActions = $env->actionSpace()->n();


//$done = false;
//echo $mo->toString($state)."\n";
//$step = 0;
//while(!$done) {
//    [$state,$reward,$done,$truncated,$info] = $env->step(rand(0,1));
//    $step++;
//}
//echo $mo->toString($state)."\n";
//echo "step=".$step."\n";
//$image2 = get_screen($la,$state,$env);

//$image = $env->render($mode='rgb_array');
//$image = $image[R(170,320+1)];
//echo $mo->toString($state,"%1.1f")."\n";
//$plt->figure();
//$plt->imshow($image,null,null,null,$origin='upper');
//$plt->figure();
//$plt->imshow($image2,null,null,null,$origin='upper');
//$plt->show();
//$env->render();
//$env->show();
//exit();

$evalEnv = new CartPoleV0($la);
$network = new QNetwork($la,$nn,$stateShape,$numActions,$convLayers,$convType,$fcLayers);
//$policy = new AnnealingEpsGreedy($la,$network,$epsStart,$epsStop,$decayRate);
$agent = new DQN(
    $la,
    network:$network,
    numActions:$numActions,
    //convLayers:$convLayers,convType:$convType,fcLayers:$fcLayers,
    epsStart:$epsStart,epsStop:$epsStop,epsDecayRate:$epsDecayRate,
    batchSize:$batchSize,gamma:$gamma,
    targetUpdatePeriod:$targetUpdatePeriod,targetUpdateTau:$targetUpdateTau,
    ddqn:$ddqn,lossFn:$lossFn,optimizerOpts:['lr'=>$learningRate],mo:$mo,
);
$agent->setCustomStateFunction($customStateFunction);
$agent->summary();

function fitplot(object $la,array $x,float $window,float $bottom) : NDArray
{
    $width = max($x)-min($x);
    if($width==0) {
        $scale = 1.0;
        $bias = $bottom;
    } else {
        $scale = $window/(max($x)-min($x));
        $bias = -min($x)*$scale+$bottom;
    }
    return $la->increment($la->scal($scale,$la->array($x)),$bias);
}

$filename = __DIR__.'\\cartpole-cnn-dqn.model';
if(!file_exists($filename)) {
    //$driver = new EpisodeRunner($la,$env,$agent,$maxExperienceSize);
    $driver = new StepRunner($la,$env,$agent,experienceSize:$maxExperienceSize,evalEnv:$evalEnv);
    $arts = [];
    //$driver->agent()->initialize();
    $history = $driver->train(numIterations:$numIterations,
        metrics:['reward','epsilon','loss','valRewards'],
        evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,logInterval:$logInterval,verbose:1
    );
    echo "\n";
    $ep = $la->array($history['iter']);
    //$arts[] = $plt->plot($ep,$la->array($history['steps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['reward']))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['loss'],200,0))[0];
    //$arts[] = $plt->plot($ep,$la->array($history['valSteps']))[0];
    $arts[] = $plt->plot($ep,$la->array($history['valRewards']))[0];
    $arts[] = $plt->plot($ep,fitplot($la,$history['epsilon'],200,0))[0];
    $plt->xlabel('Iterations');
    $plt->ylabel('Reward');
    //$plt->legend($arts,['Policy Gradient','Sarsa']);
    #$plt->legend($arts,['steps','reward','epsilon','loss','valSteps','valRewards']);
    $plt->legend($arts,['reward','loss','valRewards','epsilon']);
    //$plt->legend($arts,['steps','val_steps']);
    $plt->show();
    $agent->saveWeightsToFile($filename);
} else {
    $agent->loadWeightsFromFile($filename);
}

echo "Creating demo animation.\n";
for($i=0;$i<5;$i++) {
    echo ".";
    [$state,$info] = $env->reset();
    $state = $customStateFunction($env,$state,false);
    $env->render();
    $done=false;
    $truncated=false;
    $testReward = 0;
    $testSteps = 0;
    while(!($done||$truncated)) {
        $action = $agent->action($state,training:false,info:$info);
        [$state,$reward,$done,$truncated,$info] = $env->step($action);
        $state = $customStateFunction($env,$state,$done);
        $testReward += $reward;
        $testSteps++;
        $env->render();
    }
    $ep = $i+1;
    echo "Test Episode {$ep}, Steps: {$testSteps}, Total Reward: {$testReward}\n";
}
echo "\n";
$filename = $env->show(path:__DIR__.'\\cartpole-cnn-dqn-trained.gif');
echo "filename: {$filename}\n";
