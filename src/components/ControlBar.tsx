"use client";

import React, { useState } from "react";
import styles from "./ControlBar.module.css";

export interface ControlBarProps {
  demoCount: number;
  onDemoCountChange: (count: number) => void;
  novelty: "familiar" | "novel";
  onNoveltyChange: (novelty: "familiar" | "novel") => void;
  isAdversarial: boolean;
  onToggleAdversarial: () => void;
  className?: string;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  demoCount,
  onDemoCountChange,
  novelty,
  onNoveltyChange,
  isAdversarial,
  onToggleAdversarial,
  className = "",
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [localVal, setLocalVal] = useState(demoCount);

  // Sync with prop
  React.useEffect(() => {
    setLocalVal(demoCount);
  }, [demoCount]);

  const min = 1;
  const max = 5;
  const percentage = ((localVal - min) / (max - min)) * 100;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setLocalVal(val);
  };

  const handlePointerDown = () => {
    setIsDragging(true);
  };

  const handlePointerUp = () => {
    setIsDragging(false);
    onDemoCountChange(localVal);
  };

  const handleKeyUp = () => {
    setIsDragging(false);
    onDemoCountChange(localVal);
  };

  return (
    <div className={`${styles.controlBarContainer} ${className}`} role="toolbar" aria-label="Benchmark parameters">
      <div className={styles.leftControls}>
        {/* 1. Demo-Count Slider (1-5) */}
        <div className={styles.sliderWrapper}>
          <label htmlFor="demo-count-slider" className={styles.controlLabel}>
            Demo count:
          </label>
          <div
            className={styles.sliderTrackArea}
            onPointerDown={handlePointerDown}
            onPointerUp={handlePointerUp}
          >
            <input
              id="demo-count-slider"
              type="range"
              min={min}
              max={max}
              step={1}
              value={localVal}
              onChange={handleInputChange}
              onKeyUp={handleKeyUp}
              className={styles.rangeInput}
              aria-label="Demonstration pair count (1 to 5)"
            />
            <div className={styles.customTrack} />
            <div
              className={styles.customProgress}
              style={{ width: `${percentage}%` }}
            />
            <div
              className={styles.customThumb}
              style={{ left: `${percentage}%` }}
            />
            {isDragging && (
              <div
                className={styles.floatingTag}
                style={{ left: `${percentage}%` }}
              >
                {localVal} pair{localVal > 1 ? "s" : ""}
              </div>
            )}
          </div>
          <span className={styles.sliderValueReadout}>
            {localVal} / {max}
          </span>
        </div>

        {/* 2. Novelty Pill Switch (Familiar / Novel) */}
        <div className={styles.noveltyWrapper}>
          <span className={styles.controlLabel}>Novelty:</span>
          <div className={styles.pillSwitch} role="group" aria-label="Task novelty toggle">
            <button
              type="button"
              className={`${styles.pillOption} ${
                novelty === "familiar" ? styles.pillActive : ""
              }`}
              onClick={() => onNoveltyChange("familiar")}
              aria-pressed={novelty === "familiar"}
            >
              Familiar
            </button>
            <button
              type="button"
              className={`${styles.pillOption} ${
                novelty === "novel" ? styles.pillActive : ""
              }`}
              onClick={() => onNoveltyChange("novel")}
              aria-pressed={novelty === "novel"}
            >
              Novel
            </button>
          </div>
        </div>
      </div>

      {/* 3. "Break It" Button */}
      <button
        type="button"
        className={`${styles.breakItBtn} ${
          isAdversarial ? styles.breakItActive : ""
        }`}
        onClick={onToggleAdversarial}
        aria-pressed={isAdversarial}
      >
        {isAdversarial ? "Reset adversarial" : "Break it."}
      </button>
    </div>
  );
};
