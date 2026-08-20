<?php
declare(strict_types=1);

namespace RindowTest\RL\Agents\Unit;

use Rindow\RL\Agents\Agent\DDPG\DDPGAgent;
use Rindow\RL\Agents\Agent\SAC\SACGSDEAgent;

class ContinuousAgentTest extends MatrixTestCase
{
    public function testDDPGClipsPolicyAndNoisyActionsToActionLimit() : void
    {
        $agent = new DDPGAgent(
            $this->nn,2,2,1.5,hiddenDim:4,batchSize:1
        );
        $observation = $this->la->array([0.0,0.0]);
        $deterministic = $this->arrayOf($agent->selectActionDeterministic($observation));
        $noisy = $this->arrayOf($agent->selectAction(
            $observation,$this->la->array([100.0,-100.0])
        ));

        foreach ($deterministic as $action) {
            $this->assertGreaterThanOrEqual(-1.5,$action);
            $this->assertLessThanOrEqual(1.5,$action);
        }
        $this->assertArrayEqualsWithDelta([1.5,-1.5],$noisy);
    }

    public function testSACPolyakUpdateIsExactAndPreservesTargetArrays() : void
    {
        $agent = new SACGSDEAgent(
            $this->nn,2,1,1.0,2,4,
            1.0e-3,1.0e-3,1.0e-3,0.2,0.99,0.25,1
        );
        foreach ($agent->critic->trainableVariables() as $variable) {
            $this->la->fill(2.0,$variable->value());
        }
        foreach ($agent->criticTarget->trainableVariables() as $variable) {
            $this->la->fill(0.0,$variable->value());
        }
        $targetArrays = array_map(
            static fn($variable)=>$variable->value(),
            $agent->criticTarget->trainableVariables()
        );

        $agent->softUpdate(
            $this->nn->gradient(),$agent->critic,$agent->criticTarget,0.25
        );

        foreach ($agent->criticTarget->trainableVariables() as $i => $variable) {
            $this->assertSame($targetArrays[$i],$variable->value());
            foreach ($this->arrayOf($variable->value()->reshape([$variable->value()->size()])) as $value) {
                $this->assertEqualsWithDelta(0.5,$value,1.0e-7);
            }
        }
    }
}
