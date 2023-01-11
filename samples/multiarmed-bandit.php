<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
//use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Gym\ClassicControl\MultiarmedBandit\Slots;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\RL\Agents\Policy\EpsilonGreedy;
use Rindow\RL\Agents\Policy\AnnealingEpsGreedy;
use Rindow\RL\Agents\Network\Probabilities;
use Rindow\RL\Agents\Agent\AverageReward;
use Rindow\RL\Agents\Agent\UCB1;
use Rindow\RL\Agents\Driver\EpisodeDriver;

$mo = new MatrixOperator();
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$probabilities = [0.2, 0.4, 0.6, 0.9];
$env = new Slots($la,$probabilities);
$boltzmann = new AverageReward($la, numObs:1, numActions:count($probabilities),
    policy:new Boltzmann($la));
$egreedy = new AverageReward($la, numObs:1, numActions:count($probabilities),
    policy:new EpsilonGreedy($la,$epsilon=0.1));
$aegreedy = new AverageReward($la, numObs:1, numActions:count($probabilities),
    policy:new AnnealingEpsGreedy($la,start:$epsStart=0.9,stop:$epsEnd=0.1,decayRate:$decayRate=0.1));
$aegreedy2 = new AverageReward($la, numObs:1, numActions:count($probabilities),
    policy:new AnnealingEpsGreedy($la,decayRate:$decayRate=0.03));
$ucb1 = new UCB1($la, numActions:count($probabilities));
$driver0 = new EpisodeDriver($la,$env,$boltzmann,1);
$driver1 = new EpisodeDriver($la,$env,$egreedy,1);
$driver2 = new EpisodeDriver($la,$env,$aegreedy,1);
$driver3 = new EpisodeDriver($la,$env,$aegreedy2,1);
$driver4 = new EpisodeDriver($la,$env,$ucb1,1);

$episodes = 250;#1000;
$epochs = 1000;#50;
$dot = 100;
$arts = [];
$drivers = [$driver0,$driver1,$driver2,$driver3,$driver4];
//$drivers = [$driver0,$driver1,$driver2,$driver3];
//$drivers = [$driver0];
//$drivers = [$driver4];

foreach($drivers as $driver) {
    $avg = $la->zeros($la->alloc([$episodes]));
    for($i=0;$i<$epochs;$i++) {
        $driver->agent()->initialize();
        $driver->agent()->resetData();
        $history = $driver->train($episodes,metrics:$metrics=['reward'],
        evalInterval:$evalInterval=1,numEvalEpisodes:$numEvalEpisodes=0,verbose:$verbose=0);
        $rewards = $la->array($history['reward'],NDArray::float32);
        $la->axpy($rewards,$avg);

        if($i==0||($i+1)%10==0) {
            echo "\rEpoch ".sprintf('%4d',$i+1)." [".
                str_repeat('.',(int)ceil($i/$dot)).
                str_repeat(' ',(int)floor(($epochs-$i)/$dot))."] ";
        }

    }
    echo "\n";
    $la->scal(1.0/$epochs, $avg);
    $arts[] = $plt->plot($avg)[0];
}
foreach([0.1,0.03] as $rate) {
    $policy = new AnnealingEpsGreedy($la,$qtable,
        start:$epsStart=0.9,stop:$epsEnd=0.1,decayRate:$decayRate=$rate);
    $eps = [];
    for($i=0;$i<$episodes;$i++) {
        $eps[] = $policy->getEpsilon();
        $policy->action($dmystate=0,true);
    }
    $arts[] = $plt->plot($la->array($eps))[0];
}
$plt->legend($arts,[
    //'boltzmann',
    'boltzmann','e-greedy','ann-e-g(r=0.1)','ann-e-g(r=0.03)','UCB1',
    'eps(r=0.1)','eps(r=0.03)']);
$plt->xlabel('episodes');
$plt->ylabel('avg reward');
$plt->show();
