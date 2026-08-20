<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Rindow\RL\Agents\Agent\QLearning\QLearningAgent;
use Rindow\RL\Agents\Agent\Sarsa\TileCoder;
use Rindow\RL\Agents\Agent\Sarsa\TrueOnlineSarsaLambdaAgent;

class TabularAgentTest extends MatrixTestCase
{
    public function testTileCoderProducesOneDistinctFeaturePerTiling() : void
    {
        $coder = new TileCoder([0.0,0.0],[1.0,1.0],4,4);
        $features = $coder->encode([0.5,0.5]);
        $this->assertCount(4,$features);
        $this->assertCount(4,array_unique($features));
        $this->assertSame(100,$coder->featureCount());
        foreach ($features as $feature) {
            $this->assertGreaterThanOrEqual(0,$feature);
            $this->assertLessThan($coder->featureCount(),$feature);
        }
    }

    public function testQLearningTerminalUpdateHasExactTdErrorAndValue() : void
    {
        $coder = new TileCoder([0.0],[1.0],2,2);
        $agent = new QLearningAgent($this->la,$coder,2,learningRate:0.4,gamma:0.9);

        $delta = $agent->update([0.25],1,2.0,[0.75],true);

        $this->assertEqualsWithDelta(2.0,$delta,1.0e-7);
        // alpha per active tile is 0.2; two active weights each receive 0.4.
        $this->assertEqualsWithDelta(0.8,$agent->value([0.25],1),1.0e-7);
        $this->assertSame(1,$agent->selectActionDeterministic([0.25]));
    }

    public function testQLearningHonoursActionMask() : void
    {
        $agent = new QLearningAgent(
            $this->la,new TileCoder([0.0],[1.0],2,2),3,
            stateField:'state',actionMaskField:'mask'
        );
        $observation = ['state'=>[0.5],'mask'=>[false,false,true]];
        $this->assertSame(2,$agent->selectActionDeterministic($observation));
        $this->expectException(\InvalidArgumentException::class);
        $agent->update($observation,0,1.0,$observation,true);
    }

    public function testTrueOnlineSarsaFirstUpdateMatchesOneStepSarsa() : void
    {
        $coder = new TileCoder([0.0],[1.0],2,2);
        $agent = new TrueOnlineSarsaLambdaAgent(
            $this->la,$coder,2,learningRate:0.4,gamma:1.0,lambda:0.8
        );
        $agent->startEpisode();

        $delta = $agent->update([0.25],0,2.0,[0.75],null,true);

        $this->assertEqualsWithDelta(2.0,$delta,1.0e-7);
        $this->assertEqualsWithDelta(0.8,$agent->value([0.25],0),1.0e-7);
    }

    public function testCheckpointRoundTrip() : void
    {
        $coder = new TileCoder([0.0],[1.0],2,2);
        $source = new QLearningAgent($this->la,$coder,2,learningRate:0.4);
        $source->update([0.2],1,3.0,[0.8],true);
        $path = RINDOW_RL_TEST_TEMP_DIR.DIRECTORY_SEPARATOR.'q-learning.weights';
        $source->saveWeightsToFile($path);

        $restored = new QLearningAgent($this->la,$coder,2,learningRate:0.4);
        $restored->loadWeightsFromFile($path);
        $this->assertEqualsWithDelta(
            $source->value([0.2],1),$restored->value([0.2],1),1.0e-7
        );
    }
}
