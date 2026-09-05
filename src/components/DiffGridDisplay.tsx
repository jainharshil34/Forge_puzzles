"use client";

import React, { useState, useRef } from "react";
import styles from "./DiffGridDisplay.module.css";

export interface DiffGridDisplayProps {
  predictedGrid: number[][];
  groundTruthGrid: number[][];
  title?: string;
  cellSize?: number;
  className?: string;
  routeType: "context" | "optimization";
  incorrectCells?: [number, number][];
}

interface HoveredCell {
  row: number;
  col: number;
  predictedVal: number;
  groundTruthVal: number;
  isError: boolean;
  x: number;
  y: number;
}

export const DiffGridDisplay: React.FC<DiffGridDisplayProps> = ({
  predictedGrid,
  groundTruthGrid,
  title = "Predicted output",
  cellSize = 32,
  className = "",
  routeType,
  incorrectCells = [],
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const numRows = predictedGrid.length;
  const numCols = numRows > 0 ? predictedGrid[0].length : 0;

  // Check if a cell is marked incorrect
  const isCellError = (r: number, c: number) => {
    return incorrectCells.some(([row, col]) => row === r && col === c);
  };

  // Map ground truth values strictly to the route's accent color (1 for context teal, 2 for optimization coral)
  const formatGroundTruthValue = (val: number) => {
    if (val === 0) return 0;
    return routeType === "context" ? 1 : 2;
  };

  const handleCellMouseEnter = (
    e: React.MouseEvent<HTMLDivElement>,
    row: number,
    col: number,
    pVal: number,
    gVal: number
  ) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const cellRect = e.currentTarget.getBoundingClientRect();

    const x = cellRect.left - rect.left + cellRect.width / 2;
    const y = cellRect.top - rect.top - 6;

    setHoveredCell({
      row,
      col,
      predictedVal: pVal,
      groundTruthVal: gVal,
      isError: isCellError(row, col),
      x,
      y,
    });
  };

  const handleStackMouseEnter = () => {
    setIsHovered(true);
  };

  const handleStackMouseLeave = () => {
    setIsHovered(false);
    setHoveredCell(null);
  };

  const getCellClass = (val: number, isErr: boolean = false) => {
    let base = styles.cellVal0;
    if (val === 1) base = styles.cellVal1;
    if (val === 2) base = styles.cellVal2;

    if (isErr) {
      return `${base} ${styles.cellErrorOutline}`;
    }
    return base;
  };

  return (
    <div
      ref={containerRef}
      className={`${styles.container} ${className}`}
      style={{ position: "relative" }}
    >
      <div className={styles.header}>
        <div className={styles.titleWrapper}>
          <span className={styles.title}>{title}</span>
          <span
            className={`${styles.diffBadge} ${
              isHovered ? styles.diffBadgeActive : ""
            }`}
          >
            {isHovered
              ? "diff: ground truth"
              : incorrectCells.length > 0
              ? `${incorrectCells.length} failed cells`
              : "hover to inspect diff"}
          </span>
        </div>
        <span className={styles.dimension}>
          {numRows}×{numCols}
        </span>
      </div>

      <div
        className={styles.gridStack}
        onMouseEnter={handleStackMouseEnter}
        onMouseLeave={handleStackMouseLeave}
      >
        {/* Base Layer: Predicted Output Grid */}
        <div
          className={styles.baseGrid}
          style={{
            gridTemplateColumns: `repeat(${numCols}, ${cellSize}px)`,
            gridTemplateRows: `repeat(${numRows}, ${cellSize}px)`,
          }}
        >
          {predictedGrid.map((row, rIdx) =>
            row.map((val, cIdx) => {
              const hasError = isCellError(rIdx, cIdx);
              const gVal = groundTruthGrid[rIdx]?.[cIdx] ?? 0;
              return (
                <div
                  key={`pred-${rIdx}-${cIdx}`}
                  className={`${styles.cell} ${getCellClass(val, hasError)}`}
                  style={{ width: `${cellSize}px`, height: `${cellSize}px` }}
                  onMouseEnter={(e) =>
                    handleCellMouseEnter(e, rIdx, cIdx, val, gVal)
                  }
                  role="gridcell"
                  aria-label={`Predicted cell row ${rIdx}, col ${cIdx}, value ${val}${
                    hasError ? " (adversarial error)" : ""
                  }`}
                />
              );
            })
          )}
        </div>

        {/* Overlay Layer: Ground Truth Grid (Cross-fades in 200ms on hover) */}
        <div
          className={`${styles.overlayGrid} ${
            isHovered ? styles.overlayGridVisible : ""
          }`}
          style={{
            gridTemplateColumns: `repeat(${numCols}, ${cellSize}px)`,
            gridTemplateRows: `repeat(${numRows}, ${cellSize}px)`,
          }}
        >
          {groundTruthGrid.map((row, rIdx) =>
            row.map((rawGVal, cIdx) => {
              const mappedGVal = formatGroundTruthValue(rawGVal);
              const pVal = predictedGrid[rIdx]?.[cIdx] ?? 0;
              return (
                <div
                  key={`gt-${rIdx}-${cIdx}`}
                  className={`${styles.cell} ${getCellClass(mappedGVal)}`}
                  style={{ width: `${cellSize}px`, height: `${cellSize}px` }}
                  onMouseEnter={(e) =>
                    handleCellMouseEnter(e, rIdx, cIdx, pVal, mappedGVal)
                  }
                  data-row={rIdx}
                  data-col={cIdx}
                  data-val={mappedGVal}
                  role="gridcell"
                  aria-label={`Ground truth cell row ${rIdx}, col ${cIdx}, value ${mappedGVal}`}
                />
              );
            })
          )}
        </div>
      </div>

      {/* Neuronpedia-style Flat Monospace Tooltip */}
      {hoveredCell && (
        <div
          className={styles.tooltip}
          style={{
            left: `${hoveredCell.x}px`,
            top: `${hoveredCell.y}px`,
            transform: "translate(-50%, -100%)",
          }}
        >
          <span className={styles.tooltipLabel}>
            ({hoveredCell.row},{hoveredCell.col})
          </span>
          <span
            className={`${styles.tooltipVal} ${
              (isHovered ? hoveredCell.groundTruthVal : hoveredCell.predictedVal) === 1
                ? styles.tooltipVal1
                : (isHovered ? hoveredCell.groundTruthVal : hoveredCell.predictedVal) === 2
                ? styles.tooltipVal2
                : ""
            }`}
          >
            val: {isHovered ? hoveredCell.groundTruthVal : hoveredCell.predictedVal}
          </span>
          {hoveredCell.isError && !isHovered && (
            <span className={styles.tooltipError}>[adversarial fail]</span>
          )}
          <span className={styles.tooltipMode}>
            {isHovered ? "ground truth" : "predicted"}
          </span>
        </div>
      )}
    </div>
  );
};
