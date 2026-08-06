<?php
namespace Rindow\RL\Agents\Agent\TileCoding;

use Interop\Polite\Math\Matrix\NDArray;

/** Converts a bounded continuous observation into overlapping sparse tiles. */
class TileCoder
{
    private int $dimensions;
    private int $cellsPerTiling;
    private int $featureCount;

    /** @param float[] $low @param float[] $high */
    public function __construct(
        private array $low,
        private array $high,
        private int $numTilings = 8,
        private int $tilesPerDimension = 8,
    ) {
        $this->dimensions = count($low);
        if ($this->dimensions < 1 || count($high) !== $this->dimensions
            || $numTilings < 1 || $tilesPerDimension < 1) {
            throw new \InvalidArgumentException('Invalid tile-coder dimensions.');
        }
        foreach ($low as $i => $minimum) {
            if (!is_finite((float)$minimum) || !is_finite((float)$high[$i])
                || $minimum >= $high[$i]) {
                throw new \InvalidArgumentException('Tile-coder bounds must be finite and increasing.');
            }
        }
        // The offset can move the largest in-range value into one extra cell.
        $this->cellsPerTiling = ($tilesPerDimension + 1) ** $this->dimensions;
        $this->featureCount = $numTilings * $this->cellsPerTiling;
    }

    public function featureCount() : int { return $this->featureCount; }
    public function activeFeatureCount() : int { return $this->numTilings; }
    public function observationDimension() : int { return $this->dimensions; }

    /** @return int[] */
    public function encode(NDArray|array $observation) : array
    {
        $values = $observation instanceof NDArray ? $observation->toArray() : $observation;
        if (count($values) !== $this->dimensions) {
            throw new \InvalidArgumentException('Observation dimension does not match tile coder.');
        }
        $features = [];
        $side = $this->tilesPerDimension + 1;
        for ($tiling = 0; $tiling < $this->numTilings; $tiling++) {
            $flat = 0;
            $stride = 1;
            foreach ($values as $dimension => $value) {
                $normalized = ((float)$value - $this->low[$dimension])
                    / ($this->high[$dimension] - $this->low[$dimension]);
                $scaled = $normalized * $this->tilesPerDimension;
                // A different odd offset per dimension avoids diagonal alignment.
                $offset = $tiling * (2 * $dimension + 1) / $this->numTilings;
                $coordinate = (int)floor($scaled + $offset);
                $coordinate = max(0, min($this->tilesPerDimension, $coordinate));
                $flat += $coordinate * $stride;
                $stride *= $side;
            }
            $features[] = $tiling * $this->cellsPerTiling + $flat;
        }
        return $features;
    }
}
