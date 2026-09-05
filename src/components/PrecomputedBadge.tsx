"use client";

import React, { useState } from "react";
import styles from "./PrecomputedBadge.module.css";

export interface PrecomputedBadgeProps {
  source?: "live" | "precomputed" | string;
  className?: string;
}

export const PrecomputedBadge: React.FC<PrecomputedBadgeProps> = ({
  source = "precomputed",
  className = "",
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const isLive = source === "live";

  return (
    <div
      className={`${styles.badgeContainer} ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className={`${styles.badge} ${isLive ? styles.badgeLive : ""}`}
        tabIndex={0}
        onFocus={() => setIsHovered(true)}
        onBlur={() => setIsHovered(false)}
        aria-label={`Inference source telemetry status: ${source}`}
      >
        <span className={`${styles.dot} ${isLive ? styles.dotLive : ""}`} />
        <span>{isLive ? "LIVE" : "PRECOMPUTED"}</span>
      </div>

      {isHovered && (
        <div className={styles.tooltip} role="tooltip">
          {isLive
            ? "Inference output and loss gradients were computed dynamically on the server for this novelty/adversarial parameter set."
            : "Inference outputs and loss trajectories are loaded from offline benchmark checkpoints for instant zero-latency evaluation."}
        </div>
      )}
    </div>
  );
};
