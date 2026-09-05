"use client";

import React, { useState, useRef } from "react";
import styles from "./GridDisplay.module.css";

export interface GridDisplayProps {
  grid: number[][];
  title?: string;
  cellSize?: number;
  className?: string;
  showCoordinates?: boolean;
}

interface HoveredCell {
  row: number;
  col: number;
  value: number;
  x: number;
  y: number;
}

export const GridDisplay: React.FC<GridDisplayProps> = ({
  grid,
  title,
  cellSize = 32,
  className = "",
}) => {
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const numRows = grid.length;
  const numCols = numRows > 0 ? grid[0].length : 0;

  const handleMouseEnter = (
    e: React.MouseEvent<HTMLDivElement>,
    row: number,
    col: number,
    value: number
  ) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const cellRect = e.currentTarget.getBoundingClientRect();

    // Position tooltip just above or slightly offset to the cell
    const x = cellRect.left - rect.left + cellRect.width / 2;
    const y = cellRect.top - rect.top - 6;

    setHoveredCell({
      row,
      col,
      value,
      x,
      y,
    });
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  const getCellClass = (val: number) => {
    if (val === 1) return styles.cellVal1;
    if (val === 2) return styles.cellVal2;
    return styles.cellVal0;
  };

  return (
    <div
      ref={containerRef}
      className={`${styles.gridContainer} ${className}`}
      style={{ position: "relative" }}
    >
      {(title || numRows > 0) && (
        <div className={styles.header}>
          {title && <span className={styles.title}>{title}</span>}
          <span className={styles.dimension}>
            {numRows}×{numCols}
          </span>
        </div>
      )}

      <div
        className={styles.grid}
        style={{
          gridTemplateColumns: `repeat(${numCols}, ${cellSize}px)`,
          gridTemplateRows: `repeat(${numRows}, ${cellSize}px)`,
        }}
        onMouseLeave={handleMouseLeave}
      >
        {grid.map((row, rIdx) =>
          row.map((val, cIdx) => (
            <div
              key={`${rIdx}-${cIdx}`}
              className={`${styles.cell} ${getCellClass(val)}`}
              style={{
                width: `${cellSize}px`,
                height: `${cellSize}px`,
              }}
              onMouseEnter={(e) => handleMouseEnter(e, rIdx, cIdx, val)}
              data-row={rIdx}
              data-col={cIdx}
              data-value={val}
              role="gridcell"
              aria-label={`Cell row ${rIdx}, col ${cIdx}, value ${val}`}
            />
          ))
        )}
      </div>

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
              hoveredCell.value === 1
                ? styles.tooltipVal1
                : hoveredCell.value === 2
                ? styles.tooltipVal2
                : ""
            }`}
          >
            val: {hoveredCell.value}
          </span>
        </div>
      )}
    </div>
  );
};
