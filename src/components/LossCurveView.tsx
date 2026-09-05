"use client";

import React from "react";
import styles from "./LossCurveView.module.css";

export interface LossDataPoint {
  step: number;
  loss: number;
}

export interface LossCurveViewProps {
  lossHistory?: LossDataPoint[] | number[];
  currentStepIndex: number;
  onStepNext?: () => void;
  onReset?: () => void;
  className?: string;
}

export const LossCurveView: React.FC<LossCurveViewProps> = ({
  lossHistory = [],
  currentStepIndex,
  onStepNext,
  onReset,
  className = "",
}) => {
  // Normalize data points to { step, loss } format
  const normalizedData: LossDataPoint[] = React.useMemo(() => {
    if (!lossHistory || lossHistory.length === 0) {
      return [{ step: 0, loss: 0.89 }];
    }
    return lossHistory.map((item, idx) => {
      if (typeof item === "number") {
        return { step: idx * 10, loss: item };
      }
      return item;
    });
  }, [lossHistory]);

  const visibleData = normalizedData.slice(0, Math.max(1, currentStepIndex + 1));
  const latestPoint = visibleData[visibleData.length - 1] || { step: 0, loss: 1.0 };
  const maxStep = normalizedData[normalizedData.length - 1]?.step || 120;

  // Chart dimensions
  const width = 270;
  const height = 180;
  const paddingLeft = 32;
  const paddingRight = 16;
  const paddingTop = 16;
  const paddingBottom = 26;

  const chartW = width - paddingLeft - paddingRight;
  const chartH = height - paddingTop - paddingBottom;

  const maxLoss = 1.0;
  const minLoss = 0.0;

  const getX = (step: number) => paddingLeft + (step / (maxStep || 1)) * chartW;
  const getY = (loss: number) =>
    paddingTop + chartH - ((Math.min(maxLoss, Math.max(minLoss, loss)) - minLoss) / (maxLoss - minLoss)) * chartH;

  const pathD = visibleData
    .map((d, i) => `${i === 0 ? "M" : "L"} ${getX(d.step).toFixed(1)} ${getY(d.loss).toFixed(1)}`)
    .join(" ");

  return (
    <div className={`${styles.container} ${className}`}>
      <div className={styles.headerRow}>
        <div className={styles.titleArea}>
          <h3 className={styles.title}>Gradient loss trajectory</h3>
          <span className={styles.statusTag}>Live training</span>
        </div>
        <span className={styles.stepMetric}>
          Step {latestPoint.step} / {maxStep} (loss: {latestPoint.loss.toFixed(4)})
        </span>
      </div>

      <div className={styles.svgWrapper}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className={styles.chartSvg}
          aria-label="Optimization loss trajectory line chart"
        >
          <line
            x1={paddingLeft}
            y1={paddingTop + chartH}
            x2={width - paddingRight}
            y2={paddingTop + chartH}
            className={styles.axisLine}
          />
          <line
            x1={paddingLeft}
            y1={paddingTop}
            x2={paddingLeft}
            y2={paddingTop + chartH}
            className={styles.axisLine}
          />

          <text
            x={paddingLeft - 6}
            y={paddingTop + 4}
            textAnchor="end"
            className={styles.axisLabel}
          >
            1.0
          </text>
          <text
            x={paddingLeft - 6}
            y={paddingTop + chartH}
            textAnchor="end"
            className={styles.axisLabel}
          >
            0.0
          </text>
          <text
            x={paddingLeft}
            y={paddingTop + chartH + 16}
            textAnchor="start"
            className={styles.axisLabel}
          >
            0
          </text>
          <text
            x={width - paddingRight}
            y={paddingTop + chartH + 16}
            textAnchor="end"
            className={styles.axisLabel}
          >
            {maxStep}
          </text>

          {visibleData.length > 0 && (
            <path
              d={pathD}
              className={styles.lossLine}
            />
          )}

          {visibleData.length > 0 && (
            <circle
              cx={getX(latestPoint.step)}
              cy={getY(latestPoint.loss)}
              r={4}
              className={styles.lastPoint}
            />
          )}
        </svg>
      </div>

      <p className={styles.caption}>
        <span className={styles.captionHighlight}>Gradient descent:</span> Loss curve
        advances mechanically via sequential backpropagation steps, incurring cumulative
        latency and computational overhead per weight update.
      </p>

      <div className={styles.controlsRow}>
        <button
          type="button"
          className={styles.stepBtn}
          onClick={onStepNext}
          disabled={currentStepIndex >= normalizedData.length - 1}
        >
          {currentStepIndex >= normalizedData.length - 1
            ? "Convergence reached"
            : `Step gradient`}
        </button>
        {currentStepIndex > 0 && (
          <button
            type="button"
            className={styles.resetBtn}
            onClick={onReset}
          >
            Reset training
          </button>
        )}
      </div>
    </div>
  );
};
