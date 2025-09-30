<?php
namespace RindowTest\RL\Agents\Distribution\StateDependentNoiseDistributionTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\RL\Agents\Distribution\StateDependentNoiseDistribution;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class StateDependentNoiseDistributionTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newLa($mo)
    {
        return $mo->la();
    }

    public function newBuilder($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function std($la,NDArray $x)
    {
        // baseline
        $mean = $la->reduceMean($x,axis:0);     // ()
        $baseX = $la->add($mean,$la->copy($x),alpha:-1.0);  // (rolloutSteps)
        // std
        $n = $x->size();
        $variance = $la->scal(1/$n, $la->reduceSum($la->square($la->copy($baseX)),axis:0)); // ()
        $stdDev = $la->sqrt($variance); // ()
        return $stdDev;
    }

    public function testforwardNormal()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $N_SAMPLES = 8;
        //$N_FEATURES = 3;
        //$n_actions = 2;
        $N_FEATURES = 2;
        $n_actions = 1;

        echo "==================================================\n";
        $deterministic_actions = $g->mul($g->ones([$N_SAMPLES, $n_actions]), 0.1); // (batchSize,numActions)
        $state = $g->mul($g->ones([$N_SAMPLES, $N_FEATURES]), 0.3);                 // (batchSize,numStates)
        echo "deterministic_actions:".$la->shapeToString($deterministic_actions->shape())."\n";
        echo "state:".$la->shapeToString($state->shape())."\n";

        echo "==================================================\n";
        echo "new StateDependentNoiseDistribution(n_actions)\n";
        $dist = new StateDependentNoiseDistribution($nn, $n_actions, full_std:true, squash_output:false);

        echo "==================================================\n";
        echo "[actorModel, log_std] = probaDistributionNet(numStates)\n";
        //set_random_seed(1)
        // build model and log_std value
        [$actorModel, $log_std] = $dist->probaDistributionNet($N_FEATURES);
        $this->assertInstanceof(Module::class,$actorModel);
        echo "log_std:".$la->shapeToString($log_std->shape())."\n";
        echo "log_std=".$la->toString($log_std)."\n";

        //echo "==================================================\n";
        //// build weights from log_std and put into this instance.
        //echo "void sample_weights(log_std, batch_size)\n";
        //$dist->sample_weights($log_std, batch_size:$N_SAMPLES);

        echo "==================================================\n";
        // ganerate distribution from state and deterministic actions
        echo "dist = probaDistribution(deterministic_actions, log_std, state)\n";
        $dist = $dist->probaDistribution($deterministic_actions, $log_std, $state);

        echo "==================================================\n";
        echo "actions = dist->get_actions()\n";
        $actions = $dist->get_actions();

        echo "actions:".$la->shapeToString($actions->shape())."\n";
        echo "actions=".$la->toString($actions)."\n";
        echo "dist.mean:".$la->shapeToString($dist->distribution()->mean()->shape())."\n";
        echo "dist.mean=".$la->toString($dist->distribution()->mean())."\n";
        //$this->assertTrue($mo->la()->isclose(
        //    $g->ndarray($g->reduceMean($actions)),
        //    $g->ndarray($g->reduceMean($dist->distribution()->mean())),
        //    rtol:2e-3
        //));
        echo "std(actions):".$la->shapeToString($this->std($la,$actions)->shape())."\n";
        echo "std(actions)=".$la->toString($this->std($la,$actions))."\n";
        echo "dist.scale:".$la->shapeToString($g->reduceMean($dist->distribution()->scale(),axis:0)->shape())."\n";
        echo "dist.scale=".$la->toString($g->reduceMean($dist->distribution()->scale(),axis:0))."\n";
        $this->assertTrue($mo->la()->isclose(
            $g->ndarray($this->std($la,$actions)),
            $g->ndarray($g->reduceMean($dist->distribution()->scale(),axis:0)),
            rtol:2e-3
        ));
    }

}