<?php
namespace RindowTest\RL\Agents\Agent\PolicyGradient\PolicyTableTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\RL\Agents\Agent\PolicyGradient\PolicyTable;
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
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
            'renderer.execBackground' => true,
        ];
    }

    //public function testInitTable()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
//
    //    $plt = new Plot($this->getPlotConfig(),$mo);
//
    //    $rules = $la->array([
    //        [NAN,   1,   1],
    //        [  1, NAN,   1],
    //        [  1,   1, NAN],
    //    ]);
    //    $network = new PolicyTable($la,$rules);
    //    $states = $la->array([[0],[1],[2]]);
    //    $table = $network->table();
    //    $rules0 = $la->nan2num($la->copy($rules));
    //    $table0 = $la->nan2num($la->copy($table));
    //    $this->assertEquals($rules0->toArray(),$table0->toArray());
    //    //echo $mo->toString($table,null,true);
    //    $table1 = $la->axpy($la->copy($table),$la->copy($table),-1);
    //    $table1 = $la->nan2num($table1,1);
    //    //echo $mo->toString($table1,null,true);
    //    $this->assertEquals([[1,0,0],[0,1,0],[0,0,1]],$table1->toArray());
//
    //    $this->assertTrue(true);
    //}

    //public function testInitTable()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
//
    //    $plt = new Plot($this->getPlotConfig(),$mo);
//
    //    $rules = $la->array([
    //        [false, true,  true],
    //        [true,  false, true],
    //        [true,  true,  false],
    //    ],dtype:NDArray::bool);
    //    $network = new PolicyTable($la,$rules);
    //    $states = $la->array([[0],[1],[2]]);
    //    $table = $network->table();
    //    $rules0 = $la->masking($rules,$la->ones($la->alloc($rules->shape())));
    //    $table0 = $la->nan2num($la->copy($table));
    //    $this->assertEquals($rules0->toArray(),$table0->toArray());
    //    //echo $mo->toString($table,null,true);
    //    //$table1 = $la->axpy($la->copy($table),$la->copy($table),-1);
    //    //$table1 = $la->nan2num($table1,1);
    //    ////echo $mo->toString($table1,null,true);
    //    //$this->assertEquals([[1,0,0],[0,1,0],[0,0,1]],$table1->toArray());
//
    //    $this->assertTrue(true);
    //}

    public function testInitTable()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();

        $plt = new Plot($this->getPlotConfig(),$mo);

        $rules = $la->array([
            [false, true,  true],
            [true,  false, true],
            [true,  true,  false],
        ],dtype:NDArray::bool);
        [$numStates,$numActions] = $rules->shape();
        $network = new PolicyTable($la,$numStates,$numActions);
        //$states = $la->array([[0],[1],[2]]);
        $table = $network->table();
        //$rules0 = $la->masking($rules,$la->ones($la->alloc($rules->shape())));
        //$table0 = $la->nan2num($la->copy($table));
        //$this->assertEquals($rules0->toArray(),$table0->toArray());
        //echo $mo->toString($table,null,true);
        //$table1 = $la->axpy($la->copy($table),$la->copy($table),-1);
        //$table1 = $la->nan2num($table1,1);
        ////echo $mo->toString($table1,null,true);
        //$this->assertEquals([[1,0,0],[0,1,0],[0,0,1]],$table1->toArray());
        $this->assertEquals($la->ones($la->alloc($table->shape()))->toArray(),$table->toArray());

        $this->assertTrue(true);
    }

    //public function testSample()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
//
    //    $rules = $la->array([
    //        [NAN,NAN,  1,  1],
    //        [  1,NAN,NAN,  1],
    //        [  1,  1,NAN,NAN],
    //    ]);
    //    $network = new PolicyTable($la,$rules);
    //    $states = $la->array([[0],[1],[2]],NDArray::int32);
    //    $num = 1000;
    //    $actions = $la->alloc([$num,count($states),1],NDArray::int32);
    //    for($i=0;$i<$num;$i++) {
    //        $a = $network->sample($states);
    //        $this->assertEquals([3,1],$a->shape());
    //        $la->copy($a,$actions[$i]);
    //    }
    //    $this->assertEquals(0,$la->min($actions));
    //    $this->assertEquals(3,$la->max($actions));
    //    $avg = $la->sum($actions)/$num/count($states);
    //    $this->assertLessThan(1.6,$avg);
    //    $this->assertGreaterThan(1.4,$avg);
    //}

    //public function testProbabilities()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $rules = $la->array([
    //        [NAN,NAN,  1,  1],
    //        [  1,NAN,NAN,  1],
    //        [  1,  1,NAN,NAN],
    //        [NAN,  1,  1,NAN],
    //    ]);
    //    $network = new PolicyTable($la,$rules);
    //    $states = $la->array([[1],[0],[2]],NDArray::int32);    // (batches=3,state=1)
    //    $probs = $network->probabilities($states);
    //    $this->assertEquals([3,4],$probs->shape());
    //    $this->assertEquals([
    //        [0.5 ,0.0 ,0.0 ,0.5 ],
    //        [0.0 ,0.0 ,0.5 ,0.5 ],
    //        [0.5 ,0.5 ,0.0 ,0.0 ],
    //    ],$probs->toArray());
    //}

    //public function testProbabilities()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
    //    $rules = $la->array([
    //        [false, false, true,  true],
    //        [true,  false, false, true],
    //        [true,  true,  false, false],
    //        [false, true,  true,  false],
    //    ],dtype:NDArray::bool);
    //    $network = new PolicyTable($la,$rules);
    //    $states = $la->array([[1],[0],[2]],NDArray::int32);    // (batches=3,state=1)
    //    $probs = $network->probabilities($states);
    //    $this->assertEquals([3,4],$probs->shape());
    //    $this->assertEquals([
    //        [1, 0, 0, 1],
    //        [0, 0, 1, 1],
    //        [1, 1, 0, 0],
    //    ],$probs->toArray());
    //}

    //public function testGetQValues()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
//
    //    // network with rules
    //    $rules = $la->array([
    //        [NAN,NAN,  1,  1],
    //        [  1,NAN,NAN,  1],
    //        [  1,  1,NAN,NAN],
    //    ]);
    //    $network = new PolicyTable($la,$rules);
//
    //    //// float states
    //    //$qValues = $network->getActionValues($la->array([[0],[1],[2]],dtype:NDArray::float32));
    //    //$qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
    //    //$qValues = $la->nan2num($qValues,1);
    //    ////echo $mo->toString($qValues,null,true);
    //    //$this->assertEquals([
    //    //    [1,1,0,0],
    //    //    [0,1,1,0],
    //    //    [0,0,1,1],
    //    //],$qValues->toArray());
//
    //    // int states
    //    $qValues = $network->getActionValues($la->array([[0],[1],[2]],NDArray::int32));
    //    $qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
    //    $qValues = $la->nan2num($qValues,1);
    //    //echo $mo->toString($qValues,null,true);
    //    $this->assertEquals([
    //        [1,1,0,0],
    //        [0,1,1,0],
    //        [0,0,1,1],
    //    ],$qValues->toArray());
//
    //    // genenral
    //    $qValues = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues->shape());
    //    $qValues2 = $network->getActionValues($la->array([[1],[1],[1]],dtype:NDArray::int32));
    //    $this->assertEquals([3,4],$qValues2->shape());
    //    $qValues3 = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues3->shape());
    //    $qValues4 = $network->getActionValues($la->array([[2]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues4->shape());
    //    
    //    $qValues = $la->nan2num($la->copy($qValues))->toArray();
    //    $qValues3 = $la->nan2num($la->copy($qValues3))->toArray();
    //    $qValues4 = $la->nan2num($la->copy($qValues4))->toArray();
    //    $this->assertEquals($qValues,$qValues3);
    //    $this->assertNotEquals($qValues,$qValues4);
//
    //}

    //public function testGetQValues()
    //{
    //    $mo = $this->newMatrixOperator();
    //    $la = $mo->la();
//
    //    // network with rules
    //    $rules = $la->array([
    //        [false, false, true,  true],
    //        [true,  false, false, true],
    //        [true,  true,  false, false],
    //    ],dtype:NDArray::bool);
    //    [$numStates,$numActions] = $rules->shape();
    //    $network = new PolicyTable($la,$numStates,$numActions);
//
    //    //// float states
    //    //$qValues = $network->getActionValues($la->array([[0],[1],[2]],dtype:NDArray::float32));
    //    //$qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
    //    //$qValues = $la->nan2num($qValues,1);
    //    ////echo $mo->toString($qValues,null,true);
    //    //$this->assertEquals([
    //    //    [1,1,0,0],
    //    //    [0,1,1,0],
    //    //    [0,0,1,1],
    //    //],$qValues->toArray());
//
    //    // int states
    //    $qValues = $network->getActionValues($la->array([[0],[1],[2]],NDArray::int32));
    //    $qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
    //    $qValues = $la->nan2num($qValues,1);
    //    //echo $mo->toString($qValues,null,true);
    //    $this->assertEquals([
    //        [1,1,0,0],
    //        [0,1,1,0],
    //        [0,0,1,1],
    //    ],$qValues->toArray());
//
    //    // genenral
    //    $qValues = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues->shape());
    //    $qValues2 = $network->getActionValues($la->array([[1],[1],[1]],dtype:NDArray::int32));
    //    $this->assertEquals([3,4],$qValues2->shape());
    //    $qValues3 = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues3->shape());
    //    $qValues4 = $network->getActionValues($la->array([[2]],dtype:NDArray::int32));
    //    $this->assertEquals([1,4],$qValues4->shape());
    //    
    //    $qValues = $la->nan2num($la->copy($qValues))->toArray();
    //    $qValues3 = $la->nan2num($la->copy($qValues3))->toArray();
    //    $qValues4 = $la->nan2num($la->copy($qValues4))->toArray();
    //    $this->assertEquals($qValues,$qValues3);
    //    $this->assertNotEquals($qValues,$qValues4);
//
    //}

    public function testGetQValues()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();

        // network with rules
        $rules = $la->array([
            [false, false, true,  true],
            [true,  false, false, true],
            [true,  true,  false, false],
        ],dtype:NDArray::bool);
        [$numStates,$numActions] = $rules->shape();
        $network = new PolicyTable($la,$numStates,$numActions);
        $table = $network->table();
        $la->zeros($table);
        $la->masking($rules,$table,fill:1);

        //// float states
        //$qValues = $network->getActionValues($la->array([[0],[1],[2]],dtype:NDArray::float32));
        //$qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
        //$qValues = $la->nan2num($qValues,1);
        ////echo $mo->toString($qValues,null,true);
        //$this->assertEquals([
        //    [1,1,0,0],
        //    [0,1,1,0],
        //    [0,0,1,1],
        //],$qValues->toArray());

        // int states
        $qValues = $network->getActionValues($la->array([[0],[1],[2]],NDArray::int32));
        //$qValues = $la->axpy($la->copy($qValues),$la->copy($qValues),-1);
        //$qValues = $la->nan2num($qValues,1);
        //echo $mo->toString($qValues,null,true);
        $this->assertEquals([
            [1,1,0,0],
            [0,1,1,0],
            [0,0,1,1],
        ],$qValues->toArray());

        // genenral
        $qValues = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
        $this->assertEquals([1,4],$qValues->shape());
        $qValues2 = $network->getActionValues($la->array([[1],[1],[1]],dtype:NDArray::int32));
        $this->assertEquals([3,4],$qValues2->shape());
        $qValues3 = $network->getActionValues($la->array([[1]],dtype:NDArray::int32));
        $this->assertEquals([1,4],$qValues3->shape());
        $qValues4 = $network->getActionValues($la->array([[2]],dtype:NDArray::int32));
        $this->assertEquals([1,4],$qValues4->shape());
        
        //$qValues = $la->nan2num($la->copy($qValues))->toArray();
        //$qValues3 = $la->nan2num($la->copy($qValues3))->toArray();
        //$qValues4 = $la->nan2num($la->copy($qValues4))->toArray();
        $qValues = $qValues->toArray();
        $qValues3 = $qValues3->toArray();
        $qValues4 = $qValues4->toArray();
        $this->assertEquals($qValues,$qValues3);
        $this->assertNotEquals($qValues,$qValues4);

    }
}
