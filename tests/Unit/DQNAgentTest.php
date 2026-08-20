<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\RL\Agents\Agent\DQN\DQNAgent;
use Rindow\RL\Agents\ReplayBuffer\ReplayBuffer;

class DQNAgentTest extends MatrixTestCase
{
    private function zeroVariables(array $variables) : void
    {
        foreach ($variables as $variable) {
            $this->la->fill(0.0,$variable->value());
        }
    }

    public function testDeterministicSelectionMasksLargestQValue() : void
    {
        $agent = new DQNAgent(
            $this->nn,2,3,hiddenLayers:[4],
            stateField:'state',actionMaskField:'mask'
        );
        $this->zeroVariables($agent->qNetwork->trainableVariables());
        foreach ($agent->qNetwork->trainableVariables() as $variable) {
            if ($variable->value()->shape() === [3]) {
                $this->la->copy($this->la->array([1.0,9.0,3.0]),$variable->value());
            }
        }
        $observation = [
            'state'=>$this->la->array([0.0,0.0]),
            'mask'=>$this->la->array([true,false,true],dtype:NDArray::bool),
        ];

        $this->assertSame(2,$agent->selectActionDeterministic($observation));
    }

    public function testTerminalBellmanLossAndHardTargetSyncAreExactAndInPlace() : void
    {
        $agent = new DQNAgent(
            $this->nn,2,2,hiddenLayers:[4],learningRate:1.0e-3,
            gamma:0.9,batchSize:2,targetUpdateInterval:1
        );
        $this->zeroVariables($agent->qNetwork->trainableVariables());
        $this->zeroVariables($agent->targetNetwork->trainableVariables());
        $targetArrays = array_map(
            static fn($variable)=>$variable->value(),
            $agent->targetNetwork->trainableVariables()
        );
        $buffer = new ReplayBuffer($this->la,1,2);
        $buffer->add(
            $this->la->array([0.0,0.0]),0,2.0,
            $this->la->array([1.0,1.0]),true
        );

        $metrics = $agent->update($buffer);

        $this->assertEqualsWithDelta(4.0,$metrics['loss'],1.0e-5);
        foreach ($agent->targetNetwork->trainableVariables() as $i => $target) {
            $this->assertSame($targetArrays[$i],$target->value());
            $this->assertArrayEqualsWithDelta(
                $this->arrayOf($agent->qNetwork->trainableVariables()[$i]->value()),
                $this->arrayOf($target->value()),
                1.0e-7,
            );
        }
    }

    public function testDoubleDQNSelectsOnlineActionButEvaluatesTargetValue() : void
    {
        $makeAgent = function(bool $ddqn) : DQNAgent {
            $agent = new DQNAgent(
                $this->nn,1,2,hiddenLayers:[3],learningRate:1.0e-3,
                gamma:0.9,batchSize:1,targetUpdateInterval:100,ddqn:$ddqn
            );
            $this->zeroVariables($agent->qNetwork->trainableVariables());
            $this->zeroVariables($agent->targetNetwork->trainableVariables());
            foreach ($agent->qNetwork->trainableVariables() as $variable) {
                if ($variable->value()->shape() === [2]) {
                    // Online selects action 0, while current transition uses action 1 (Q=1).
                    $this->la->copy($this->la->array([5.0,1.0]),$variable->value());
                }
            }
            foreach ($agent->targetNetwork->trainableVariables() as $variable) {
                if ($variable->value()->shape() === [2]) {
                    // Target itself prefers action 1.
                    $this->la->copy($this->la->array([2.0,4.0]),$variable->value());
                }
            }
            return $agent;
        };
        $buffer = new ReplayBuffer($this->la,1,1);
        $buffer->add($this->la->array([0.0]),1,0.0,$this->la->array([0.0]),false);

        $dqnLoss = $makeAgent(false)->update($buffer)['loss'];
        $ddqnLoss = $makeAgent(true)->update($buffer)['loss'];

        // DQN target=0.9*4 => (1-3.6)^2=6.76
        // DDQN selects online action 0 and evaluates target Q=2 => (1-1.8)^2=0.64
        $this->assertEqualsWithDelta(6.76,$dqnLoss,1.0e-4);
        $this->assertEqualsWithDelta(0.64,$ddqnLoss,1.0e-4);
    }
}
