<?php
namespace Rindow\RL\Agents\Util;

use Interop\Polite\Math\Matrix\NDArray;

class GradientClipping
{
    /**
     * @param array<NDArray> $gradients
     * @return array<NDArray>
     */
    public static function clipByGlobalNorm(
        object $la,
        array $gradients,
        float $maxNorm,
    ) : array {
        if (is_infinite($maxNorm) || count($gradients)===0) {
            return $gradients;
        }

        if (!$la->accelerated() || $maxNorm<=0.0) {
            return self::clipOnHost($la,$gradients,$maxNorm);
        }

        $norms = $la->alloc(
            [count($gradients)],
            dtype:$gradients[0]->dtype(),
        );
        $blas = $la->getBlas();
        $queue = $la->getQueue();
        foreach ($gradients as $i => $gradient) {
            // Write every norm into one device vector.  Calling la->nrm2()
            // without an output would allocate one scalar NDArray per gradient.
            $blas->nrm2(
                $gradient->size(),
                $norms->buffer(),$norms->offset()+$i,
                $gradient->buffer(),$gradient->offset(),1,
                $queue,
            );
        }

        $la->square($norms);
        $sumSquares = $la->alloc([],dtype:$norms->dtype());
        $math = $la->getMath();
        $math->sum(
            $norms->size(),
            $sumSquares->buffer(),$sumSquares->offset(),
            $norms->buffer(),$norms->offset(),1,
            $queue,
        );

        // scale = maxNorm / max(globalNorm,maxNorm)
        $la->sqrt($sumSquares);
        $la->maximum($sumSquares,$maxNorm);
        $la->reciprocal($sumSquares);
        $la->scal($maxNorm,$sumSquares);
        foreach ($gradients as $gradient) {
            // A scalar-shaped X is broadcast over A by multiply().
            $la->multiply($sumSquares,$gradient);
        }
        return $gradients;
    }

    /**
     * @param array<NDArray> $gradients
     * @return array<NDArray>
     */
    private static function clipOnHost(
        object $la,
        array $gradients,
        float $maxNorm,
    ) : array {
        $sumSquares = 0.0;
        foreach ($gradients as $gradient) {
            $gradientNorm = $la->nrm2($gradient);
            if ($gradientNorm instanceof NDArray) {
                $gradientNorm = $la->scalar($gradientNorm);
            }
            $sumSquares += (float)$gradientNorm*(float)$gradientNorm;
        }
        $norm = sqrt($sumSquares);
        if ($norm<=$maxNorm || $norm==0.0) {
            return $gradients;
        }
        $scale = $maxNorm/($norm+1.0e-8);
        foreach ($gradients as $gradient) {
            $la->scal($scale,$gradient);
        }
        return $gradients;
    }
}
