<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent\A2C\A2CAgent;
use Rindow\RL\Agents\Agent\PPO\PPOAgent;

class ActorCriticAgentTest extends MatrixTestCase
{
    private function assertFiniteMetrics(array $metrics) : void
    {
        foreach ($metrics as $name => $value) {
            $this->assertIsFloat($value,$name);
            $this->assertTrue(is_finite($value),$name.' must be finite');
        }
    }

    public function testA2CDiscreteUpdateReturnsFiniteLossesAndExpectedEntropy() : void
    {
        $agent = new A2CAgent(
            $this->nn,2,3,hiddenLayers:[4],learningRate:1.0e-3,
            entropyWeight:0.0,maxGradNorm:10.0
        );
        foreach ($agent->network->trainableVariables() as $variable) {
            $this->la->fill(0.0,$variable->value());
        }
        $rollout = [
            $this->la->array([[0.0,0.0],[1.0,0.0]],dtype:NDArray::float32),
            $this->la->array([0,1],dtype:NDArray::int32),
            $this->la->array([1.0,-1.0],dtype:NDArray::float32),
            $this->la->array([1.0,1.0],dtype:NDArray::float32),
        ];

        $metrics = $agent->update($rollout);

        $this->assertFiniteMetrics($metrics);
        $this->assertEqualsWithDelta(log(3.0),$metrics['entropy'],1.0e-5);
        $this->assertGreaterThan(0.0,$metrics['value_loss']);
    }

    public function testPPOOneEpochUpdateReturnsFiniteClippedObjectiveMetrics() : void
    {
        $agent = new PPOAgent(
            $this->nn,2,3,hiddenLayers:[4],learningRate:1.0e-3,
            entropyWeight:0.0,epochs:1,batchSize:2,maxGradNorm:10.0,
            clipValueLoss:false
        );
        foreach ($agent->network->trainableVariables() as $variable) {
            $this->la->fill(0.0,$variable->value());
        }
        $rollout = [
            $this->la->array([[0.0,0.0],[1.0,0.0]],dtype:NDArray::float32),
            $this->la->array([0,1],dtype:NDArray::int32),
            $this->la->array([log(1.0/3.0),log(1.0/3.0)],dtype:NDArray::float32),
            $this->la->array([1.0,-1.0],dtype:NDArray::float32),
            $this->la->array([1.0,1.0],dtype:NDArray::float32),
            $this->la->array([0.0,0.0],dtype:NDArray::float32),
        ];

        $metrics = $agent->update($rollout);

        $this->assertFiniteMetrics($metrics);
        $this->assertEqualsWithDelta(log(3.0),$metrics['entropy'],1.0e-5);
        $this->assertGreaterThan(0.0,$metrics['value_loss']);
    }

    public function testA2CAndPPODeterministicPoliciesRespectMask() : void
    {
        foreach ([
            new A2CAgent($this->nn,2,3,hiddenLayers:[4],stateField:'state',actionMaskField:'mask'),
            new PPOAgent($this->nn,2,3,hiddenLayers:[4],stateField:'state',actionMaskField:'mask'),
        ] as $agent) {
            foreach ($agent->network->trainableVariables() as $variable) {
                $this->la->fill(0.0,$variable->value());
            }
            $observation = [
                'state'=>$this->la->array([0.0,0.0]),
                'mask'=>$this->la->array([false,true,false],dtype:NDArray::bool),
            ];
            $this->assertSame(1,$agent->selectActionDeterministic($observation));
        }
    }
}
