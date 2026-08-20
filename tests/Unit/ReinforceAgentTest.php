<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent\Reinforce\ReinforceAgent;

class ReinforceAgentTest extends MatrixTestCase
{
    public function testUniformPolicyHasExactMonteCarloLossAndEntropy() : void
    {
        $agent = new ReinforceAgent(
            $this->nn,2,3,hiddenLayers:[4],learningRate:1.0e-3,
            entropyWeight:0.0,maxGradNorm:10.0
        );
        foreach ($agent->network->trainableVariables() as $variable) {
            $this->la->fill(0.0,$variable->value());
        }

        $metrics = $agent->update(
            $this->la->array([[0.0,0.0],[1.0,0.0]],dtype:NDArray::float32),
            $this->la->array([0,1],dtype:NDArray::int32),
            $this->la->array([2.0,2.0],dtype:NDArray::float32),
        );

        $this->assertEqualsWithDelta(2.0*log(3.0),$metrics['policy_loss'],1.0e-5);
        $this->assertEqualsWithDelta(log(3.0),$metrics['entropy'],1.0e-5);
    }

    public function testDeterministicActionUsesLargestLogit() : void
    {
        $agent = new ReinforceAgent($this->nn,2,3,hiddenLayers:[4]);
        foreach ($agent->network->trainableVariables() as $variable) {
            $this->la->fill(0.0,$variable->value());
            if ($variable->value()->shape() === [3]) {
                $this->la->copy($this->la->array([-1.0,4.0,2.0]),$variable->value());
            }
        }
        $this->assertSame(1,$agent->selectActionDeterministic($this->la->array([0,0])));
    }
}
