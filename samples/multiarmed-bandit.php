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
use Rindow\RL\Agents\Agent\PolicyGradient\PolicyGradient;
use Rindow\RL\Agents\Agent\AverageReward\AverageReward;
use Rindow\RL\Agents\Agent\AverageReward\ValueTable;
use Rindow\RL\Agents\Agent\UCB1\UCB1;
use Rindow\RL\Agents\Runner\EpisodeRunner;

$mo = new MatrixOperator();
$la = $mo->la();
//$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$probabilities = [0.2, 0.4, 0.6, 0.9];
$stateFunc = function($env,$x,$done) use ($la) {
    return $la->expandDims($x,axis:-1);
};
$env = new Slots($la,$probabilities);
$slotRules = $la->ones($la->alloc([1,count($probabilities)]));
$boltzmann = new PolicyGradient($la, numStates:1, numActions:count($probabilities), eta:0.1,
    policy:new Boltzmann($la,fromLogits:true));
$egreedy = new AverageReward($la, numActions:count($probabilities),
    policy:new EpsilonGreedy($la,$epsilon=0.1));
$aegreedy = new AverageReward($la, numActions:count($probabilities),
    policy:new AnnealingEpsGreedy($la,start:$epsStart=0.9,stop:$epsEnd=0.1,decayRate:$decayRate=0.1));
$aegreedy2 = new AverageReward($la, numActions:count($probabilities),
    policy:new AnnealingEpsGreedy($la,decayRate:$decayRate=0.03));
$ucb1 = new UCB1($la, numActions:count($probabilities));
$driver0 = new EpisodeRunner($la,$env,$boltzmann,1);
$driver1 = new EpisodeRunner($la,$env,$egreedy,1);
$driver2 = new EpisodeRunner($la,$env,$aegreedy,1);
$driver3 = new EpisodeRunner($la,$env,$aegreedy2,1);
$driver4 = new EpisodeRunner($la,$env,$ucb1,1);
$boltzmann->setCustomStateFunction($stateFunc);
$egreedy->setCustomStateFunction($stateFunc);
$aegreedy->setCustomStateFunction($stateFunc);
$aegreedy2->setCustomStateFunction($stateFunc);
$ucb1->setCustomStateFunction($stateFunc);

$episodes = 250;#1000;
$epochs = 1000;#1000;#50;
$dot = 100;
$arts = [];
$drivers = [$driver0,$driver1,$driver2,$driver3,$driver4];
//$drivers = [$driver0,$driver1,$driver2,$driver3];
//$drivers = [$driver0];
//$drivers = [$driver1];
//$drivers = [$driver2];
//$drivers = [$driver3];
//$drivers = [$driver4];

foreach($drivers as $driver) {
    $avg = $la->zeros($la->alloc([$episodes]));
    for($i=0;$i<$epochs;$i++) {
        $driver->initialize();
        $driver->agent()->initialize();
        $driver->agent()->resetData();
        $history = $driver->train(
            $episodes,metrics:$metrics=['reward'],
            evalInterval:1,numEvalEpisodes:0,verbose:0
        );
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
$dmyTable = new ValueTable($la,numActions:count($probabilities));
foreach([0.1,0.03] as $rate) {
    $policy = new AnnealingEpsGreedy($la,
        start:$epsStart=0.9,stop:$epsEnd=0.1,decayRate:$decayRate=$rate);
    $eps = [];
    for($i=0;$i<$episodes;$i++) {
        $eps[] = $policy->getEpsilon();
        $policy->actions($dmyTable,$mo->zeros([1,1],dtype:NDArray::int32),training:true,masks:null);
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
