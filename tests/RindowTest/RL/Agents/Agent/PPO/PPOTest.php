<?php
namespace RindowTest\RL\Agents\Agent\PPO\PPOTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\RL\Agents\Policy;
use Rindow\RL\Agents\Network;
use Rindow\RL\Agents\Estimator;
use Rindow\RL\Agents\EventManager;
use Rindow\RL\Agents\Agent\PPO\PPO;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;
use Rindow\RL\Agents\Policy\Boltzmann;
use Rindow\RL\Agents\Util\Metrics;
use Rindow\RL\Gym\Core\Spaces\Box;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;

class TestPolicy implements Policy
{
    public function __construct($fixedAction)
    {
        $this->fixedAction = $fixedAction;
    }

    public function isContinuousActions() : bool
    {
        return false;
    }

    public function register(?EventManager $eventManager=null) : void
    {}

    public function initialize() : void // : Operation
    {}

    public function actions(Estimator $network, NDArray $values, bool $training, ?NDArray $masks) : NDArray
    {
        return $this->fixedAction;
    }
}

class TESTPPOClass extends PPO
{
    public function test_compute_advantages_and_returns(
        NDArray $rewards, // (rolloutSteps)
        NDArray $values,  // (rolloutSteps+1,1)
        array $dones,
        float $gamma,
        float $gaeLambda,
        ) : array
    {
        return $this->compute_advantages_and_returns(
            $rewards,
            $values,
            $dones,
            $gamma,
            $gaeLambda,
        );
    }
    public function test_standardize(
        NDArray $x,         // (rolloutSteps)
        ?bool $ddof=null
        ) : NDArray
    {
        return $this->standardize($x,$ddof);
    }

    public function test_clip_by_global_norm(
        array $arrayList, float $clipNorm
        ) : array
    {
        return $this->clip_by_global_norm($arrayList,$clipNorm);
    }
}

class PPOTest extends TestCase
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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    public function testActionOnBoltzmann()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $policy = new Boltzmann($la);

        $agent = new PPO($la,
            policy:$policy,
            nn:$nn, stateShape:[1], numActions:2,fcLayers:[100]);
        $states = [
            $la->array([0]),
            $la->array([1]),
        ];
        for($i=0;$i<100;$i++) {
            $actions = $agent->action($states,training:true);
            $this->assertEquals([2],$actions->shape());
            $this->assertEquals(NDArray::int32,$actions->dtype());
        }
    }

    public function testGAE()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $agent = new TESTPPOClass($la,
            batchSize:3,epochs:4,rolloutSteps:3,
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100],
            normAdv:true,
        );

        $rewards = [-1.0856306552886963, 0.9973454475402832, 0.28297850489616394, -1.5062947273254395, -0.5786002278327942, 1.6514365673065186, -2.4266791343688965, -0.4289126396179199, 1.2659361362457275, -0.8667404055595398];
        $values = [-0.2049688994884491, 0.33364400267601013, -0.26434439420700073, -0.13481280207633972, 0.23233819007873535, 1.0453966856002808, 0.3585890233516693, -1.134916067123413, -1.349687933921814, -0.27974411845207214, -0.8566373586654663];
        $dones = [False, False, False, False, True, False, False, False, False, False];
        $gamma = 0.9;
        $gaeLambda = 0.95;
        $rewards = $la->array($rewards);
        $values = $la->expandDims($la->array($values),axis:-1);
        [$advantages, $returns] = $agent->test_compute_advantages_and_returns($rewards,$values,$dones,$gamma,$gaeLambda);

        $trues_advantages= [-1.0648003, -0.5665709, -1.1606578, -1.8557299, -0.8109384,
                            -1.9460603, -3.3623745,  0.51967,    1.2027901, -1.3579699];
        $trues_returns=    [-1.2697692, -0.23292688,-1.4250021, -1.9905428, -0.5786002,
                            -0.9006636, -3.0037856, -0.61524606,-0.1468978, -1.637714 ];
        $trues_advantages = $la->array($trues_advantages);
        $trues_returns = $la->array($trues_returns);

        $this->assertTrue($la->isclose($trues_advantages,$advantages));
        $this->assertTrue($la->isclose($trues_returns,$returns));
    }

    public function testStandardize()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $agent = new TESTPPOClass($la,
            batchSize:3,epochs:4,rolloutSteps:3,
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100],
            normAdv:true,
        );

        $data_array = [ 68.22012, -0.01524353, 28.89531, 92.59357, 74.4759, -27.854034, -43.14157, -33.886078, -0.8924103, 15.548714 ];
        $data_array = $la->array($data_array);
        $results = $agent->test_standardize($data_array);

        $trues = [ 1.1195153,  -0.38347518,  0.25332487,  1.6563785,   1.2573085,
                  -0.996668,   -1.3333999,  -1.1295332,  -0.40279615, -0.04065474];
        $trues = $la->array($trues);
        $this->assertTrue($la->isclose($trues,$results));
    }

    public function testClipByGlobalNorm()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $agent = new TESTPPOClass($la,
            batchSize:3,epochs:4,rolloutSteps:3,
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100],
            normAdv:true,
        );

        // Pytorchからコピーした入力データ
        $grad1_data = [[1.5409960746765137, -0.293428897857666, -2.1787893772125244], [0.5684312582015991, -1.0845223665237427, -1.3985954523086548]];
        $grad2_data = [0.40334683656692505, 0.8380263447761536, -0.7192575931549072, -0.40334352850914, -0.5966353416442871];
        $grad3_data = [[0.18203648924827576, -0.8566746115684509, 1.1006041765213013, -1.0711873769760132]];


        $grad1 = $la->array($grad1_data, NDArray::float32);
        $grad2 = $la->array($grad2_data, NDArray::float32);
        $grad3 = $la->array($grad3_data, NDArray::float32);
        $grads = [$grad1, $grad2, $grad3];

        $clip_norm_value = 0.5;

        //echo "\n--- Case 1: Norm is LARGER than clip_norm ({$clip_norm_value}) ---\n";
        $scaled_grads_large = [$la->scal(10, $la->copy($grad1)), $la->scal(10, $la->copy($grad2)), $la->scal(10, $la->copy($grad3))];

        //echo "grad1 sum: ".$la->sum($scaled_grads_large[0])."\n";
        //echo "grad2 sum: ".$la->sum($scaled_grads_large[1])."\n";
        //echo "grad3 sum: ".$la->sum($scaled_grads_large[2])."\n";
        $this->assertTrue($la->isclose($la->array(-28.45908546447754),$la->array($la->sum($scaled_grads_large[0]))));
        $this->assertTrue($la->isclose($la->array(-4.778633117675781),$la->array($la->sum($scaled_grads_large[1]))));
        $this->assertTrue($la->isclose($la->array(-6.452213287353516),$la->array($la->sum($scaled_grads_large[2]))));

        // PHP版の関数を実行
        $clipped_grads_large = $agent->test_clip_by_global_norm($scaled_grads_large, $clip_norm_value);

        // グローバルノルムを計算して比較
        $beforeGlobalNorm = 0.0;
        foreach($scaled_grads_large as $array) { $beforeGlobalNorm += ($la->nrm2($array)**2); }
        $beforeGlobalNorm = sqrt($beforeGlobalNorm);

        $afterGlobalNorm = 0.0;
        foreach($clipped_grads_large as $array) { $afterGlobalNorm += ($la->nrm2($array)**2); }
        $afterGlobalNorm = sqrt($afterGlobalNorm);

        //echo "\n[PHP results for Case 1]\n";
        //echo "Global norm (before clipping): {$beforeGlobalNorm}\n";
        //echo "Global norm (after clipping): {$afterGlobalNorm}\n";
        //echo "grad1_clipped sum: ".$la->sum($clipped_grads_large[0])."\n";
        //echo "grad2_clipped sum: ".$la->sum($clipped_grads_large[1])."\n";
        //echo "grad3_clipped sum: ".$la->sum($clipped_grads_large[2])."\n";
        $this->assertTrue($la->isclose($la->array(39.61064910888672),$la->array($beforeGlobalNorm)));
        $this->assertTrue($la->isclose($la->array($clip_norm_value),$la->array($afterGlobalNorm)));
        $this->assertTrue($la->isclose($la->array(-0.3592352569103241),$la->array($la->sum($clipped_grads_large[0]))));
        $this->assertTrue($la->isclose($la->array(-0.06032004952430725),$la->array($la->sum($clipped_grads_large[1]))));
        $this->assertTrue($la->isclose($la->array(-0.08144544064998627),$la->array($la->sum($clipped_grads_large[2]))));


        //echo "\n--- Case 2: Norm is SMALLER than clip_norm ({$clip_norm_value}) ---\n";
        $scaled_grads_small = [$la->scal(0.01, $la->copy($grad1)), $la->scal(0.01, $la->copy($grad2)), $la->scal(0.01, $la->copy($grad3))];

        //echo "grad1 sum: ".$la->sum($scaled_grads_small[0])."\n";
        //echo "grad2 sum: ".$la->sum($scaled_grads_small[1])."\n";
        //echo "grad3 sum: ".$la->sum($scaled_grads_small[2])."\n";
        $this->assertTrue($la->isclose($la->array(-0.028459087014198303),$la->array($la->sum($scaled_grads_small[0]))));
        $this->assertTrue($la->isclose($la->array(-0.004778632894158363),$la->array($la->sum($scaled_grads_small[1]))));
        $this->assertTrue($la->isclose($la->array(-0.006452213041484356),$la->array($la->sum($scaled_grads_small[2]))));

        $clipped_grads_small = $agent->test_clip_by_global_norm($scaled_grads_small, $clip_norm_value);

        $globalNormSmall = 0.0;
        foreach($scaled_grads_small as $array) { $globalNormSmall += ($la->nrm2($array)**2); }
        $globalNormSmall = sqrt($globalNormSmall);

        $afterGlobalNorm = 0.0;
        foreach($clipped_grads_small as $array) { $afterGlobalNorm += ($la->nrm2($array)**2); }
        $afterGlobalNorm = sqrt($afterGlobalNorm);

        //echo "\n[PHP results for Case 2]\n";
        //echo "Global norm (before clipping): {$globalNormSmall}\n";
        //echo "Global norm (after clipping): {$afterGlobalNorm}\n";
        //echo "grad1_clipped sum: ".$la->sum($clipped_grads_small[0])."\n";
        //echo "grad2_clipped sum: ".$la->sum($clipped_grads_small[1])."\n";
        //echo "grad3_clipped sum: ".$la->sum($clipped_grads_small[2])."\n";
        $this->assertTrue($la->isclose($la->array(0.0396106466650962),$la->array($globalNormSmall)));
        $this->assertTrue($la->isclose($la->array(0.0396106466650962),$la->array($afterGlobalNorm)));
        $this->assertTrue($la->isclose($la->array(-0.028459087014198303),$la->array($la->sum($clipped_grads_small[0]))));
        $this->assertTrue($la->isclose($la->array(-0.004778632894158363),$la->array($la->sum($clipped_grads_small[1]))));
        $this->assertTrue($la->isclose($la->array(-0.006452213041484356),$la->array($la->sum($clipped_grads_small[2]))));
    }

    public function testUpdate()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $agent = new PPO($la,
            batchSize:3,epochs:4,rolloutSteps:3,
            nn:$nn, stateShape:[1], numActions:2, fcLayers:[100],
            normAdv:true,
        );
        $metrics = new Metrics();
        $agent->setMetrics($metrics);
        $metrics->attract(['loss','entropy']);
        $mem = new ReplayBuffer($la,$maxsize=3);
        //[$state,$action,$nextState,$reward,$done,$info]
        $losses = [];
        for($i=0;$i<100;$i++) {
            $mem->add([$la->array([0]),$la->array(1,dtype:NDArray::int32),$la->array([1]),1,false,false,[]]);
            $mem->add([$la->array([1]),$la->array(1,dtype:NDArray::int32),$la->array([2]),1,false,false,[]]);
            $mem->add([$la->array([2]),$la->array(0,dtype:NDArray::int32),$la->array([3]),1,false,false,[]]);
            $losses[] = $agent->update($mem);
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('PPO');
        $plt->show();
        $this->assertTrue(true);
    }

    public function testActionContinuous()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);

        $actionSpace = new Box($la,high:1,low:-1,shape:[2]);
        $agent = new PPO($la,
            continuous:true,
            nn:$nn, stateShape:[1], actionSpace:$actionSpace, fcLayers:[100],
        );
        $states = [
            $la->array([0]),
            $la->array([1]),
        ];
        for($i=0;$i<100;$i++) {
            $actions = $agent->action($states,training:true);
            $this->assertEquals([2,2],$actions->shape());
            $this->assertEquals(NDArray::float32,$actions->dtype());
        }
    }

    public function testUpdateContinuous()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $nn = $this->newBuilder($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $actionSpace = new Box($la,high:1,low:-1,shape:[2]);
        $agent = new PPO($la,
            batchSize:3,epochs:4,rolloutSteps:3,
            nn:$nn, stateShape:[1], actionSpace:$actionSpace, fcLayers:[100],
            normAdv:true,
            continuous:true,
        );
        $metrics = new Metrics();
        $agent->setMetrics($metrics);
        $metrics->attract(['loss','entropy']);
        $mem = new ReplayBuffer($la,$maxsize=3);
        //[$state,$action,$nextState,$reward,$done,$info]
        $losses = [];
        for($i=0;$i<100;$i++) {
            $mem->add([$la->array([0]),$la->array([1,1]),$la->array([1]),1,false,false,[]]);
            $mem->add([$la->array([1]),$la->array([1,1]),$la->array([2]),1,false,false,[]]);
            $mem->add([$la->array([2]),$la->array([0,0]),$la->array([3]),1,false,false,[]]);
            $losses[] = $agent->update($mem);
        }
        $losses = $la->array($losses);
        $plt->plot($losses);
        $plt->legend(['losses']);
        $plt->title('PPO');
        $plt->show();
        $this->assertTrue(true);
    }
}
