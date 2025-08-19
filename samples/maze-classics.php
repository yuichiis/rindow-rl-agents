<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\RL\Agents\Runner\EpisodeRunner;
use Rindow\RL\Agents\Agent\PolicyGradient\PolicyGradient;
use Rindow\RL\Agents\Agent\Sarsa\Sarsa;
use Rindow\RL\Agents\Agent\QLearning\QLearning;
use Rindow\RL\Agents\Agent\QLearning\QTable;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$mazeRules = $la->array([
//   UP    DOWN  RIGHT LEFT
    [false, true, true,false], // 0  +-+-+-+
    [false, true, true, true], // 1  |0 1 2|
    [false,false,false, true], // 2  + + +-+
    [true,  true,false,false], // 3  |3|4 5|
    [true, false, true,false], // 4  + +-+ +
    [false, true,false, true], // 5  |6 7|8|
    [true, false, true,false], // 6  +-+-+-+
    [false,false,false, true], // 7
    [true, false,false,false], // 8
],dtype:NDArray::bool);
[$width,$height,$exit] = [3,3,8];
$stateFunc = function($env,$obs,$done) use ($la,$width) {
    $location = $obs['location'];
    $y = $location[0];
    $x = $location[1];
    $pos = $y*$width + $x;
    $pos = $la->array([$pos],dtype:NDArray::int32);
    $mask = $obs['actionMask'];
    return ['location'=>$pos,'actionMask'=>$mask];
};
$stateField = 'location';
$maxEpisodeSteps = 100;
$episodes = 100;#250;#15;#
$epochs = 50; # 500; #
$evalInterval=1;
$numEvalEpisodes=10;
$dot = (int)($epochs/10);

$env = new Maze($la,$mazeRules,$width,$height,$exit,maxEpisodeSteps:$maxEpisodeSteps);
$evalEnv = new Maze($la,$mazeRules,$width,$height,$exit,maxEpisodeSteps:$maxEpisodeSteps);
$numStates = $width*$height;
$numActions = $env->actionSpace()->n();

$policy = new AnnealingEpsGreedy($la,start:1.0,stop:0.01,decayRate:0.05,episodeAnnealing:true);
//$policy = new AnnealingEpsGreedy($la,$valueTable,$espstart=0.1,$stop=0.1,$decayRate=0.01);
$pg = new PolicyGradient($la,$numStates,$numActions,eta:0.1,stateField:$stateField,mo:$mo);
$sarsa = new Sarsa($la,$numStates,$numActions,$policy,eta:0.1,gamma:0.9,stateField:$stateField,mo:$mo);
$qlearning = new QLearning($la,$numStates,$numActions,$policy,eta:0.1,gamma:0.9,stateField:$stateField,mo:$mo);
$pg->setCustomStateFunction($stateFunc);
$sarsa->setCustomStateFunction($stateFunc);
$qlearning->setCustomStateFunction($stateFunc);

$driver1 = new EpisodeRunner($la,$env,$pg,experienceSize:10000,evalEnv:$evalEnv);
$driver2 = new EpisodeRunner($la,$env,$sarsa,experienceSize:2,evalEnv:$evalEnv);
$driver3 = new EpisodeRunner($la,$env,$qlearning,experienceSize:2,evalEnv:$evalEnv);
$drivers = [$driver1,$driver2,$driver3];
//$drivers = [$driver2];

$arts = [];
foreach ($drivers as $driver) {
    echo get_class($driver->agent())."\n";
    $avgsteps = $la->zeros($la->alloc([(int)floor($episodes/$evalInterval)]));
    $avgvalsteps = $la->zeros($la->alloc([(int)floor($episodes/$evalInterval)]));
    for($i=0;$i<$epochs;$i++) {
        $driver->initialize();
        $driver->agent()->initialize();
        $driver->agent()->resetData();
        $history = $driver->train(
            numIterations:$episodes,metrics:['steps','valSteps','epsilon'],
            evalInterval:$evalInterval,numEvalEpisodes:$numEvalEpisodes,verbose:0);
        $stepslog = $la->array($history['steps'],NDArray::float32);
        $la->axpy($stepslog,$avgsteps);
        $stepslog = $la->array($history['valSteps'],NDArray::float32);
        $la->axpy($stepslog,$avgvalsteps);

        echo "\rEpoch ".sprintf('%2d',$i+1)." [".
            str_repeat('.',(int)ceil($i/$dot)).
            str_repeat(' ',(int)floor(($epochs-$i)/$dot))."] ";
    }
    echo "\n";
    $la->scal(1.0/$epochs, $avgsteps);
    $la->scal(1.0/$epochs, $avgvalsteps);
    $ep = $mo->arange((int)floor($episodes/$evalInterval),$evalInterval,$evalInterval);
    $arts[] = $plt->plot($ep,$avgsteps)[0];
    $arts[] = $plt->plot($ep,$avgvalsteps)[0];
}
$arts[] = $plt->plot($ep,$la->scal($maxEpisodeSteps,$la->array($history['epsilon'])))[0];
//$plt->legend($arts,['Policy Gradient']);
//$plt->legend($arts,['Sarsa','Q-Learing']);
$plt->legend($arts,['Policy Gradient','Policy Gradient(val)',
                    'Sarsa','Sarsa(val)','Q-Learing','Q-Learing(val)','epsilon']);
$plt->xlabel('episodes');
$plt->ylabel('avg steps');
$plt->show();
