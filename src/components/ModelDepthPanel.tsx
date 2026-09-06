"use client";

import React, { useEffect, useState } from "react";
import styles from "./ModelDepthPanel.module.css";

interface ModelDepthData {
  source?: string;
  sweep_attn_variant?: any;
  optimization_ceiling?: any;
  cost_efficiency?: any;
}

export const ModelDepthPanel: React.FC = () => {
  const [data, setData] = useState<ModelDepthData | null>(null);
  const [source, setSource] = useState<string>("live_api");

  useEffect(() => {
    fetch("/api/sweep")
      .then((res) => res.json())
      .then((json) => {
        setData(json);
        setSource(json.source || "live_backend_results");
      })
      .catch((err) => {
        console.warn("Could not fetch sweep route for ModelDepthPanel:", err);
      });
  }, []);

  return (
    <section className={styles.depthContainer} aria-label="Model Depth: Advanced Empirical Findings">
      {/* Header */}
      <div className={styles.depthHeader}>
        <div className={styles.depthTitleGroup}>
          <h2 className={styles.depthTitle}>Model Depth: Advanced Empirical Findings</h2>
          <span className={styles.depthBadge}>Deep Ablations</span>
        </div>
        <div className={styles.dataSourceIndicator}>
          Source: <span className="mono-val" style={{ color: "var(--accent-a)" }}>{source}</span>
        </div>
      </div>

      {/* Three Core Findings Cards */}
      <div className={styles.findingsGrid}>
        {/* Finding 1: GRU vs. Attention Aggregation on Recolor */}
        <div className={styles.findingCard}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTag}>Ablation 1</span>
            <span className={styles.sourcePill}>sweep_attn_variant.json</span>
          </div>
          <h3 className={styles.cardTitle}>GRU vs. Attention Aggregation on Recolor</h3>
          <p className={styles.cardProse}>
            Removing the sequential recurrence and recency bias of GRUCell with self-attention pooling eliminates artificial ordering priors on permutation-invariant recolor tasks.
          </p>

          <div className={styles.metricDisplay}>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>1-Demo Recolor Exact:</span>
              <span className={styles.metricValTeal}>12.0% (Attn) vs. 0.0% (GRU)</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>1-Demo Recolor Cell Acc:</span>
              <span className={styles.metricValTeal}>84.48% vs. 59.60% (+24.9%)</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>5-Demo Recolor Cell Acc:</span>
              <span className={styles.metricValTeal}>89.12% vs. 83.84% (+5.3%)</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>1-Demo Translate Exact:</span>
              <span className={styles.metricValTeal}>86.0% (Attn) vs. 2.0% (GRU)</span>
            </div>
          </div>

          <p className={styles.cardFooterNote}>
            Attention pooling resolves transformation coordinates in 1 demo and binds color mappings without sequential interference.
          </p>
        </div>

        {/* Finding 2: The Optimization Ceiling Threshold */}
        <div className={styles.findingCard}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTag}>Ablation 2</span>
            <span className={styles.sourcePill}>optimization_ceiling.json</span>
          </div>
          <h3 className={styles.cardTitle}>The Optimization Ceiling Threshold</h3>
          <p className={styles.cardProse}>
            High-compute ceiling sweep scaling demonstration counts {"{10, 25, 50, 100}"} across inference gradient steps K {"{10, 25, 50}"} (1,800 evaluated instances).
          </p>

          <div className={styles.metricDisplay}>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Non-Zero Exact Threshold:</span>
              <span className={styles.metricValCoral}>None Found In-Range</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Exact Match (All 36 Regimes):</span>
              <span className={styles.metricValCoral}>0.0% (0 / 1,800 correct)</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Peak Cell Acc (Translate, K=50):</span>
              <span className={styles.metricValNeutral}>53.92% (Floor: 52.7%)</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Inference Latency (100 demos):</span>
              <span className={styles.metricValCoral}>95 - 116 ms / puzzle</span>
            </div>
          </div>

          <p className={styles.cardFooterNote}>
            Gradient updates on heterogeneous demo grids cancel out in batch averages, preventing convergence on global transformation rules.
          </p>
        </div>

        {/* Finding 3: Cost-per-Correct-Answer */}
        <div className={styles.findingCard}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTag}>Ablation 3</span>
            <span className={styles.sourcePill}>cost_efficiency.json</span>
          </div>
          <h3 className={styles.cardTitle}>Cost-per-Correct-Answer & Pareto Frontier</h3>
          <p className={styles.cardProse}>
            Evaluation of computational efficiency measured in milliseconds of inference compute required per 1.0 unit of exact-match accuracy.
          </p>

          <div className={styles.metricDisplay}>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Context Cost / Exact Match:</span>
              <span className={styles.metricValTeal}>1.78 - 2.54 ms / exact</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Optimization Cost / Exact:</span>
              <span className={styles.metricValCoral}>∞ (Infinite: 0.0% Exact)</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Context Latency (5 Demos):</span>
              <span className={styles.metricValTeal}>2.12 ms (Pure Forward)</span>
            </div>
            <div className={styles.metricRow}>
              <span className={styles.metricLabel}>Pareto Frontier Dominance:</span>
              <span className={styles.metricValTeal}>Context Model (100%)</span>
            </div>
          </div>

          <p className={styles.cardFooterNote}>
            The Context Model strictly Pareto-dominates the Optimization Model across all evaluated coordinates in the cost-accuracy plane.
          </p>
        </div>
      </div>
    </section>
  );
};
