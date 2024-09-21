<?php
namespace RindowTest\RL\Agents\Network\PolicyTableTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Agents\Network\PolicyTable;
use Rindow\Math\Plot\Plot;
use LogicException;
use InvalidArgumentException;
use Throwable;


class PolicyTableTest extends TestCase
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
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    public function testInitTable()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();

        $plt = new Plot($this->getPlotConfig(),$mo);

        $rules = $la->array([
            [NAN,   1,   1],
            [  1, NAN,   1],
            [  1,   1, NAN],
        ]);
        $network = new PolicyTable($la,$rules);
        $states = $la->array([[0],[1],[2]]);
        $table = $network->table();
        $rules0 = $la->nan2num($la->copy($rules));
        $table0 = $la->nan2num($la->copy($table));
        $this->assertEquals($rules0->toArray(),$table0->toArray());
        //echo $mo->toString($table,null,true);
        $table1 = $la->axpy($la->copy($table),$la->copy($table),-1);
        $table1 = $la->nan2num($table1,1);
        //echo $mo->toString($table1,null,true);
        $this->assertEquals([[1,0,0],[0,1,0],[0,0,1]],$table1->toArray());

        $this->assertTrue(true);
    }

    public function testSample()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();

        $rules = $la->array([
            [NAN,NAN,  1,  1],
            [  1,NAN,NAN,  1],
            [  1,  1,NAN,NAN],
        ]);
        $network = new PolicyTable($la,$rules);
        $obs = $la->array([[0],[1],[2]],NDArray::uint32);
        $num = 1000;
        $actions = $la->alloc([$num,count($obs),1],NDArray::uint32);
        for($i=0;$i<$num;$i++) {
            $a = $network->sample($obs);
            $this->assertEquals([3,1],$a->shape());
            $la->copy($a,$actions[$i]);
        }
        $this->assertEquals(0,$la->min($actions));
        $this->assertEquals(3,$la->max($actions));
        $avg = $la->sum($actions)/$num/count($obs);
        $this->assertLessThan(1.6,$avg);
        $this->assertGreaterThan(1.4,$avg);
    }

    public function testGetQValues()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();

        // network with rules
        $rules = $la->array([
            [NAN,NAN,  1,  1],
            [  1,NAN,NAN,  1],
            [  1,  1,NAN,NAN],
        ]);
        $network = new PolicyTable($la,$rules);

        // float states
        $qValues = $network->getQValues($la->array([[0.0],[1.0],[2.0]]));
        $qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
        $qValues = $la->nan2num($qValues,1);
        //echo $mo->toString($qValues,null,true);
        $this->assertEquals([
            [1,1,0,0],
            [0,1,1,0],
            [0,0,1,1],
        ],$qValues->toArray());

        // int states
        $qValues = $network->getQValues($la->array([[0],[1],[2]],NDArray::uint32));
        $qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
        $qValues = $la->nan2num($qValues,1);
        //echo $mo->toString($qValues,null,true);
        $this->assertEquals([
            [1,1,0,0],
            [0,1,1,0],
            [0,0,1,1],
        ],$qValues->toArray());

        // genenral
        $qValues = $network->getQValues($la->array([[1.0]]));
        $this->assertEquals([1,4],$qValues->shape());
        $qValues2 = $network->getQValues($la->array([[1.0],[1.0],[1.0]]));
        $this->assertEquals([3,4],$qValues2->shape());
        $qValues3 = $network->getQValues($la->array([[1.0]]));
        $this->assertEquals([1,4],$qValues3->shape());
        $qValues4 = $network->getQValues($la->array([[2.0]]));
        $this->assertEquals([1,4],$qValues4->shape());
        
        $qValues = $la->nan2num($la->copy($qValues))->toArray();
        $qValues3 = $la->nan2num($la->copy($qValues3))->toArray();
        $qValues4 = $la->nan2num($la->copy($qValues4))->toArray();
        $this->assertEquals($qValues,$qValues3);
        $this->assertNotEquals($qValues,$qValues4);

    }
}
