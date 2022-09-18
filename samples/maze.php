<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\Maze\Maze;
use Rindow\RL\Agents\Driver\EpisodeDriver;
use Rindow\RL\Agents\Agent\PolicyGradient;
use Rindow\RL\Agents\Agent\Sarsa;
use Rindow\RL\Agents\Agent\QLearning;
use Rindow\RL\Agents\Agent\DQN;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\Network\QTable;
use Rindow\RL\Agents\Network\QNetwork;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$mazeRules = $la->array([
//   UP    DOWN  RIGHT LEFT
    [NAN,    1,    1,  NAN], // 0  +-+-+-+
    [NAN,    1,    1,    1], // 1  |0 1 2|
    [NAN,  NAN,  NAN,    1], // 2  + + +-+
    [  1,    1,  NAN,  NAN], // 3  |3|4 5|
    [  1,  NAN,    1,  NAN], // 4  + +-+ +
    [NAN,    1,  NAN,    1], // 5  |6 7|8|
    [  1,  NAN,    1,  NAN], // 6  +-+-+-+
    [NAN,  NAN,  NAN,    1], // 7
    [  1,  NAN,  NAN,  NAN], // 8
]);
[$width,$height,$exit] = [3,3,8];
$env = new Maze($la,$mazeRules,$width,$height,$exit,$throw=true,$maxEpisodeSteps=100);

$qtable = new QTable($la,$mazeRules);
//$policy = new AnnealingEpsGreedy($la,$qtable,$espstart=1.0,$stop=0.01,$decayRate=0.01);
//$policy = new AnnealingEpsGreedy($la,$qtable,$espstart=0.1,$stop=0.1,$decayRate=0.01);
$pg = new PolicyGradient($la,$mazeRules,$eta=0.1,$mo);
$sarsa = new Sarsa($la,$qtable,$policy,$eta=0.1,$gamma=0.9,$mo);
//$qlearning = new QLearning($la,$qtable,$policy,$eta=0.1,$gamma=0.9,$mo);
//$network = new QNetwork($la,$nn,$obsSize=[1],$numActions=4,$convLayers=null,$convType=null,$fcLayers=[100],$activation='tanh',null,$mazeRules);
//$policy2 = new AnnealingEpsGreedy($la,$network,$epsStart=0.9,$epsStop=0.1,$decayRate=0.07);
//$dqn = new Dqn(
//    $la,$network,$policy2,
//    $batchSize=32,$gamma=0.9,$targetUpdatePeriod=5,$targetUpdateTau=0.05,
//    $ddqn=true,null,$lossFn=$nn->losses->MeanSquaredError(),null,null,['lr'=>$learningRate=1e-3],
//    null,null,null,null,null,null,$mo
//);
//$dqn->summary();

$driver1 = new EpisodeDriver($la,$env,$pg,$experienceSize=10000);
$driver2 = new EpisodeDriver($la,$env,$sarsa,$experienceSize=2,null,$episodeAnnealing=true);
//$driver3 = new EpisodeDriver($la,$env,$qlearning,$experienceSize=2,null,$episodeAnnealing=true);
//$driver4 = new EpisodeDriver($la,$env,$dqn,$experienceSize=10000,null,$episodeAnnealing=true);
//$drivers = [$driver1,$driver2,$driver3];
$drivers = [$driver2];

$episodes = 100;#250;#15;#
$epochs = 50;#500;#
$evalInterval=1;
$numEvalEpisodes=10;
$dot = (int)($epochs/10);
$arts = [];
foreach ($drivers as $driver) {
    $avgsteps = $la->zeros($la->alloc([(int)floor($episodes/$evalInterval)]));
    $avgvalsteps = $la->zeros($la->alloc([(int)floor($episodes/$evalInterval)]));
    for($i=0;$i<$epochs;$i++) {
        $driver->agent()->initialize();
        $history = $driver->train(
            $episodes,null,$metrics=['steps','val_steps','epsilon'],
            $evalInterval,$numEvalEpisodes,null,$verbose=0);
        $stepslog = $la->array($history['steps'],NDArray::float32);
        $la->axpy($stepslog,$avgsteps);
        $stepslog = $la->array($history['val_steps'],NDArray::float32);
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
