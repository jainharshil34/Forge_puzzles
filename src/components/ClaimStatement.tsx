"use client";

import React from "react";
import styles from "./ClaimStatement.module.css";

export const ClaimStatement: React.FC = () => {
  return (
    <section className={styles.claimContainer} aria-label="Core Empirical Claim">
      <div className={styles.claimHeader}>
        <span className={styles.claimLabel}>Core Empirical Claim</span>
        <span className={styles.claimBadge}>Verified Benchmark</span>
      </div>
      <p className={styles.claimProse}>
        <span className={styles.highlightCoral}>Gradient-adapted</span> architectures attempt to resolve novel task rules by backpropagating loss over parametric weights at inference time - an O(K) backward-pass procedure that incurs latency cost, risks interference with previously consolidated parameters, and, under a <span className={styles.monoNumber}>5</span>-demonstration budget, fails to recover the underlying rule at any tested step count up to K=<span className={styles.monoNumber}>10</span> (<span className={styles.monoNumber}>0%</span> exact-match) - recovering only once demonstrations scale to <span className={styles.monoNumber}>none found in-range (0% exact across 10-100 demos, K=10-50)</span> - whereas a fixed-dimensional <span className={styles.highlightTeal}>recurrent state</span> accumulates task-specific structure through gated forward-pass integration of demonstration pairs, achieving <span className={styles.monoNumber}>68.7% ± 3.5%</span> exact-match by five demonstrations across 5 seeds, <span className={styles.monoNumber}>16.0%</span> on the previously weak recolor rule family after attention-based aggregation, and no measurable degradation of prior task competence (<span className={styles.monoNumber}>0.0000</span> forgetting), though this adaptation does not extend to rule parameters or rule families absent from training.
      </p>
    </section>
  );
};
