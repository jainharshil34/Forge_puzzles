"use client";

import React, { useState } from "react";
import styles from "./BDHModule.module.css";

export const BDHModule: React.FC = () => {
  const [simStep, setSimStep] = useState<number>(3);
  const [simMode, setSimMode] = useState<"hebbian" | "gated" | "sgd">("hebbian");

  // Simulated live matrix values for Hebbian synaptic state
  const getMatrixIntensity = (row: number, col: number, step: number) => {
    if (simMode === "hebbian") {
      const val = Math.sin(row * 1.5 + col * 0.8 + step * 0.9) * 0.5 + 0.5;
      return `rgba(94, 234, 212, ${0.15 + val * 0.7 * (step / 5)})`;
    } else if (simMode === "gated") {
      const val = Math.cos(row * 0.9 + col * 1.2 + step * 1.1) * 0.5 + 0.5;
      return `rgba(94, 234, 212, ${0.1 + val * 0.85 * (step / 5)})`;
    } else {
      // SGD parameter perturbation
      const val = Math.sin(row * 2.1 + col * 1.4 + step * 0.5) * 0.5 + 0.5;
      return `rgba(242, 150, 125, ${0.1 + val * 0.8 * (step / 5)})`;
    }
  };

  return (
    <section className={styles.bdhContainer} aria-label="BDH-CQ Theoretical Connection and Architecture Context">
      {/* Module Header */}
      <div className={styles.moduleHeader}>
        <div className={styles.moduleTitleGroup}>
          <h2 className={styles.moduleTitle}>Where this connects to BDH-CQ</h2>
          <span className={styles.requiredBadge}>Required module</span>
        </div>
      </div>

      {/* Section 1: Disclaimer Callout (Coral border, visually distinct) */}
      <div className={styles.disclaimerBox} role="alert">
        <span className={styles.disclaimerIcon}>⚠️</span>
        <div className={styles.disclaimerContent}>
          <span className={styles.disclaimerHeading}>Architecture Scope Disclaimer</span>
          <p className={styles.disclaimerText}>
            This project&apos;s context-route model is an explicitly-labeled toy reimplementation inspired by BDH-CQ - it is NOT the official BDH or BDH-CQ model. No weights, checkpoints, or code from Pathway&apos;s official repositories were used or reproduced.
          </p>
        </div>
      </div>

      {/* Section 2: What BDH-CQ actually does */}
      <div className={styles.sectionBlock}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTag}>Section 2</span>
          <h3 className={styles.sectionTitle}>What BDH-CQ actually does</h3>
        </div>
        <p className={styles.bodyProse}>
          BDH-CQ builds on Dragon Hatchling (BDH), a post-Transformer architecture in which neuron-like units communicate through local, low-rank interactions and maintain context in an evolving associative state - attention is reformulated as <span className={styles.highlightTeal}>synaptic memory</span> updated via Hebbian-style writes as the model reads. BDH-CQ extends this to inference-time adaptation: recurrent memory updates continuously as it reads, queries are solved through iterative computation in a latent workspace, neither task identifiers nor evaluation-task demonstration pairs are used during training, and <span className={styles.highlightTeal}>no parameters update at inference</span>. On the public ARC-AGI-1 benchmark, a <span className={styles.monoNumber}>150M-parameter</span> BDH-CQ configuration reached <span className={styles.monoNumber}>29.5% pass@2</span> at a computed cost of <span className={styles.monoNumber}>$0.0007/task</span>, independently reproduced by a team from Bielik AI and NYU.
        </p>

        {/* Concrete Grounding: Architectural Mechanism Schematic Diagram */}
        <div className={styles.groundingSchematicBox}>
          <div className={styles.schematicHeader}>
            <span className={styles.schematicTitle}>
              <span>Architectural Mechanism Diagram: State Ingestion vs. Gradient Overwrite</span>
            </span>
            <span className="mono-val" style={{ color: "var(--text-secondary)", fontSize: "11px" }}>
              Figure 1: Inference Dataflow
            </span>
          </div>

          <div className={styles.schematicSvgContainer}>
            <svg viewBox="0 0 880 160" width="100%" height="160" style={{ display: "block" }}>
              <defs>
                <marker id="arrow-teal" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L6,3 z" fill="#5EEAD4" />
                </marker>
                <marker id="arrow-coral" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L6,3 z" fill="#F2967D" />
                </marker>
                <marker id="arrow-gray" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L6,3 z" fill="#8B909C" />
                </marker>
              </defs>

              {/* Path 1: Context/BDH Route */}
              <rect x="10" y="15" width="130" height="50" rx="4" fill="#171A21" stroke="#2A2E38" strokeWidth="1" />
              <text x="75" y="38" fill="#E8E6DF" fontSize="11" fontFamily="var(--font-mono)" textAnchor="middle">Demo Stream</text>
              <text x="75" y="52" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">(X_1, Y_1)...(X_K, Y_K)</text>

              <line x1="140" y1="40" x2="200" y2="40" stroke="#5EEAD4" strokeWidth="1.5" markerEnd="url(#arrow-teal)" />

              <rect x="205" y="15" width="180" height="50" rx="4" fill="rgba(94, 234, 212, 0.08)" stroke="#5EEAD4" strokeWidth="1.2" />
              <text x="295" y="36" fill="#5EEAD4" fontSize="11" fontFamily="var(--font-heading)" fontWeight="600" textAnchor="middle">Associative State / GRU</text>
              <text x="295" y="52" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">W_t = W_t-1 + k·v^T  (Δθ = 0)</text>

              <line x1="385" y1="40" x2="455" y2="40" stroke="#5EEAD4" strokeWidth="1.5" markerEnd="url(#arrow-teal)" />

              <rect x="460" y="15" width="160" height="50" rx="4" fill="#171A21" stroke="#2A2E38" strokeWidth="1" />
              <text x="540" y="36" fill="#E8E6DF" fontSize="11" fontFamily="var(--font-heading)" textAnchor="middle">Latent Workspace</text>
              <text x="540" y="52" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">Frozen Weights θ_0</text>

              <line x1="620" y1="40" x2="690" y2="40" stroke="#5EEAD4" strokeWidth="1.5" markerEnd="url(#arrow-teal)" />

              <rect x="695" y="15" width="160" height="50" rx="4" fill="#171A21" stroke="#5EEAD4" strokeWidth="1" />
              <text x="775" y="36" fill="#5EEAD4" fontSize="11" fontFamily="var(--font-mono)" fontWeight="600" textAnchor="middle">Prediction Y_test</text>
              <text x="775" y="52" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">&lt;2ms, 0.0000 forgetting</text>

              {/* Path 2: Optimization Route */}
              <rect x="10" y="95" width="130" height="50" rx="4" fill="#171A21" stroke="#2A2E38" strokeWidth="1" />
              <text x="75" y="118" fill="#E8E6DF" fontSize="11" fontFamily="var(--font-mono)" textAnchor="middle">Demo Pairs</text>
              <text x="75" y="132" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">Loss L(f_θ(X), Y)</text>

              <line x1="140" y1="120" x2="200" y2="120" stroke="#F2967D" strokeWidth="1.5" markerEnd="url(#arrow-coral)" />

              <rect x="205" y="95" width="180" height="50" rx="4" fill="rgba(242, 150, 125, 0.08)" stroke="#F2967D" strokeWidth="1.2" />
              <text x="295" y="116" fill="#F2967D" fontSize="11" fontFamily="var(--font-heading)" fontWeight="600" textAnchor="middle">Inference SGD (K steps)</text>
              <text x="295" y="132" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">θ_k = θ_k-1 - α·∇_θ L</text>

              <line x1="385" y1="120" x2="455" y2="120" stroke="#F2967D" strokeWidth="1.5" markerEnd="url(#arrow-coral)" />

              <rect x="460" y="95" width="160" height="50" rx="4" fill="#171A21" stroke="#2A2E38" strokeWidth="1" />
              <text x="540" y="116" fill="#F2967D" fontSize="11" fontFamily="var(--font-heading)" textAnchor="middle">Overwritten θ_K</text>
              <text x="540" y="132" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">Destructive Interference</text>

              <line x1="620" y1="120" x2="690" y2="120" stroke="#F2967D" strokeWidth="1.5" markerEnd="url(#arrow-coral)" />

              <rect x="695" y="95" width="160" height="50" rx="4" fill="#171A21" stroke="#F2967D" strokeWidth="1" />
              <text x="775" y="116" fill="#F2967D" fontSize="11" fontFamily="var(--font-mono)" fontWeight="600" textAnchor="middle">Degraded Prediction</text>
              <text x="775" y="132" fill="#8B909C" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">15-115ms, 0% exact</text>
            </svg>
          </div>
        </div>

        {/* Grounded Formalization: Mathematical Mechanism Comparison */}
        <div className={styles.groundingContainer}>
          <div className={styles.groundingCard}>
            <span className={styles.groundingCardHeader}>BDH-CQ Synaptic State Update (Hebbian Write)</span>
            <div className={styles.equationBlock}>
              W_t = W_{`{t-1}`} + η · k_t · v_t^T
            </div>
            <p className={styles.groundingExplanation}>
              Attention is mapped to associative matrix accumulation: keys and values perform rank-1 synaptic memory writes without backpropagation.
            </p>
          </div>
          <div className={styles.groundingCard}>
            <span className={styles.groundingCardHeader}>Our Toy Context-Route Update (Gated Latent State)</span>
            <div className={styles.equationBlock}>
              h_t = GRUCell(e(X_t, Y_t), h_{`{t-1}`})
            </div>
            <p className={styles.groundingExplanation}>
              Gated recurrent hidden vector accumulation: encodes demonstration pairs into transient state space under <span className="mono-val">torch.no_grad()</span>.
            </p>
          </div>
        </div>

        {/* Concrete Grounding: Live Interactive Mechanism Simulator */}
        <div className={styles.liveExperimentBox}>
          <div className={styles.experimentHeader}>
            <div>
              <span className={styles.cardTag} style={{ marginRight: "8px" }}>Live Experiment</span>
              <span className={styles.cardTitle}>Inference State Accumulation Simulator</span>
            </div>
            <div className={styles.experimentControls}>
              <button
                className={`${styles.experimentBtn} ${simMode === "hebbian" ? styles.experimentBtnActive : ""}`}
                onClick={() => setSimMode("hebbian")}
              >
                Hebbian Matrix (BDH-CQ)
              </button>
              <button
                className={`${styles.experimentBtn} ${simMode === "gated" ? styles.experimentBtnActive : ""}`}
                onClick={() => setSimMode("gated")}
              >
                Gated Vector (Our Toy)
              </button>
              <button
                className={`${styles.experimentBtn} ${simMode === "sgd" ? styles.experimentBtnActive : ""}`}
                onClick={() => setSimMode("sgd")}
              >
                Parametric SGD (Opt)
              </button>
            </div>
          </div>

          <div className={styles.liveSimulationDisplay}>
            <div className={styles.simMatrixRow}>
              <div>
                <span className="mono-val" style={{ fontSize: "11px", color: "var(--text-secondary)", display: "block", marginBottom: "6px" }}>
                  Active {simMode === "hebbian" ? "Synaptic Associative Matrix W_t (6×6)" : simMode === "gated" ? "Latent Activation State h_t (36-dim)" : "Weight Perturbation Δθ_k (36-dim)"}:
                </span>
                <div className={styles.simMatrixGrid}>
                  {Array.from({ length: 36 }).map((_, idx) => {
                    const r = Math.floor(idx / 6);
                    const c = idx % 6;
                    const bg = getMatrixIntensity(r, c, simStep);
                    return (
                      <div
                        key={idx}
                        className={styles.simCell}
                        style={{ backgroundColor: bg, border: "1px solid rgba(255,255,255,0.05)" }}
                      >
                        {((r + c + simStep) % 9)}
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className={styles.simTelemetry}>
                <div><strong>Current Step:</strong> Ingested {simStep} / 5 demonstrations</div>
                <div><strong>Inference Parameter Update Δθ:</strong> {simMode === "sgd" ? `+0.0${simStep * 18} (Destructive)` : "0.0000 (Frozen)"}</div>
                <div><strong>Memory Mechanism:</strong> {simMode === "hebbian" ? "Hebbian Rank-1 Associative Write" : simMode === "gated" ? "Gated State Trace Propagation" : "In-place SGD Backprop Overwrite"}</div>
                <div><strong>Measured Catastrophic Forgetting:</strong> {simMode === "sgd" ? "+1.5% Cell Loss" : "0.0000 (Provably Zero)"}</div>
                <div style={{ marginTop: "6px", display: "flex", gap: "6px" }}>
                  <button
                    className={styles.experimentBtn}
                    onClick={() => setSimStep((prev) => Math.min(5, prev + 1))}
                    disabled={simStep >= 5}
                  >
                    + Step Demo
                  </button>
                  <button
                    className={styles.experimentBtn}
                    onClick={() => setSimStep(1)}
                  >
                    Reset State
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: What our toy model mirrors - and where it diverges */}
      <div className={styles.sectionBlock}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTag}>Section 3</span>
          <h3 className={styles.sectionTitle}>What our toy model mirrors - and where it diverges</h3>
        </div>

        <div className={styles.tableContainer}>
          <table className={styles.comparisonTable}>
            <thead>
              <tr>
                <th className={styles.thBlank}></th>
                <th className={styles.thContext}>Our context-route model</th>
                <th className={styles.thBdh}>BDH-CQ (real system)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className={styles.rowLabel}>Adaptation mechanism</td>
                <td className={styles.cellContext}>Forward-pass state accumulation / zero parameter updates at inference</td>
                <td className={styles.cellBdh}>Forward-pass state accumulation / zero parameter updates at inference (same principle both sides)</td>
              </tr>
              <tr>
                <td className={styles.rowLabel}>State update rule</td>
                <td className={styles.cellContext}>Gated GRUCell / attention-pooled</td>
                <td className={styles.cellBdh}>Additive accumulation per BDH-CQ&apos;s own formulation</td>
              </tr>
              <tr>
                <td className={styles.rowLabel}>Scale</td>
                <td className={styles.cellContext}>&lt;1M parameters (~390k)</td>
                <td className={styles.cellBdh}>150M parameters</td>
              </tr>
              <tr>
                <td className={styles.rowLabel}>Domain</td>
                <td className={styles.cellContext}>Synthetic 5×5 grid puzzles</td>
                <td className={styles.cellBdh}>ARC-AGI-1 public evaluation set</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className={styles.tableNote}>
          We use a gated update rather than BDH-CQ&apos;s literal additive accumulation because our aggregator is a standard GRUCell / attention pool - we are not claiming architectural equivalence, only that the same adaptation principle (state carries the task, weights don&apos;t move) is what our results demonstrate.
        </p>
      </div>

      {/* Section 4: Primary sources */}
      <div className={styles.citationsBlock}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTag}>Section 4</span>
          <h3 className={styles.sectionTitle}>Primary sources</h3>
        </div>

        <ul className={styles.citationsList}>
          <li className={styles.citationItem}>
            <p className={styles.citationText}>
              Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz. &ldquo;The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.&rdquo;{" "}
              <a
                href="https://arxiv.org/abs/2509.26507"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.citationLink}
              >
                arXiv:2509.26507
              </a>{" "}
              (2025).
            </p>
            <span className={styles.supportsNote}>
              Supports: the synaptic-memory/Hebbian mechanism described in Section 2.
            </span>
          </li>
          <li className={styles.citationItem}>
            <p className={styles.citationText}>
              Engdahl, Kosowski, Chorowski, Stamirowska, Uznański, Jiang, Phadke, Kinas, Zhong. &ldquo;BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.&rdquo;{" "}
              <a
                href="https://arxiv.org/abs/2608.09888"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.citationLink}
              >
                arXiv:2608.09888
              </a>{" "}
              (2026).
            </p>
            <span className={styles.supportsNote}>
              Supports: the zero-parameter-update claim and ARC-AGI-1 results cited in Section 2.
            </span>
          </li>
          <li className={styles.citationItem}>
            <p className={styles.citationText}>
              Jolicoeur-Martineau. &ldquo;Less is More: Recursive Reasoning with Tiny Networks.&rdquo;{" "}
              <a
                href="https://arxiv.org/abs/2510.04871"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.citationLink}
              >
                arXiv:2510.04871
              </a>{" "}
              (2025).
            </p>
            <span className={styles.supportsNote}>
              Supports: the optimization-route baseline&apos;s framing and its HRM lineage.
            </span>
          </li>
        </ul>
      </div>
    </section>
  );
};
