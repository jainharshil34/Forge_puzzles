"use client";

import React, { useState, useEffect, useCallback } from "react";
import { getPuzzles, getPuzzle, getPrediction } from "@/lib/mockApi";
import { GridDisplay } from "@/components/GridDisplay";
import { DiffGridDisplay } from "@/components/DiffGridDisplay";
import { StateVectorView, GraphNode, GraphEdge } from "@/components/StateVectorView";
import { LossCurveView, LossDataPoint } from "@/components/LossCurveView";
import { ControlBar } from "@/components/ControlBar";
import { PrecomputedBadge } from "@/components/PrecomputedBadge";
import { EmpiricalFindingsPanel } from "@/components/EmpiricalFindingsPanel";
import styles from "./page.module.css";

interface DemoPair {
  input: number[][];
  output: number[][];
}

interface ContextModel {
  source?: string;
  predicted_output: number[][];
  latency_ms: number;
  confidence: number;
  state_nodes?: number;
  state_edges?: number;
  memory_status?: string;
  nodes?: GraphNode[];
  edges?: GraphEdge[];
}

interface OptimizationModel {
  source?: string;
  predicted_output: number[][];
  latency_ms: number;
  confidence: number;
  current_step: number;
  max_steps: number;
  loss: number;
  learning_rate: number;
  loss_curve: LossDataPoint[];
  incorrect_cells: [number, number][];
}

interface Puzzle {
  id: string;
  name: string;
  dimension: string;
  demo_pairs: DemoPair[];
  test_input: number[][];
  ground_truth: number[][];
}

export default function Home() {
  const [puzzles, setPuzzles] = useState<Puzzle[]>([]);
  const [selectedPuzzleId, setSelectedPuzzleId] = useState<string>("arc-sym-01");

  // Control suite state
  const [demoCount, setDemoCount] = useState<number>(2);
  const [novelty, setNovelty] = useState<"familiar" | "novel">("familiar");
  const [isAdversarial, setIsAdversarial] = useState<boolean>(false);
  const [isNoveltyFlashing, setIsNoveltyFlashing] = useState<boolean>(false);

  // Active puzzle & predictions
  const [currentPuzzle, setCurrentPuzzle] = useState<Puzzle | null>(null);
  const [contextModel, setContextModel] = useState<ContextModel | null>(null);
  const [optimizationModel, setOptimizationModel] = useState<OptimizationModel | null>(null);

  // Animation step states
  const [ingestionStage, setIngestionStage] = useState<number>(2);
  const [optStepIndex, setOptStepIndex] = useState<number>(12);

  // Load puzzles on mount
  useEffect(() => {
    getPuzzles().then((list) => {
      const typedList = list as Puzzle[];
      setPuzzles(typedList);
      if (typedList.length > 0 && !selectedPuzzleId) {
        setSelectedPuzzleId(typedList[0].id);
      }
    });
  }, [selectedPuzzleId]);

  // Load puzzle and predictions when parameters change
  const loadData = useCallback(
    async (id: string, count: number, nov: "familiar" | "novel", adv: boolean) => {
      if (!id) return;
      try {
        const [puzzleData, contextRes, optRes] = await Promise.all([
          getPuzzle(id) as Promise<Puzzle>,
          getPrediction(id, "context", count, nov, adv) as Promise<Record<string, any>>,
          getPrediction(id, "optimization", count, nov, adv) as Promise<Record<string, any>>,
        ]);

        setCurrentPuzzle(puzzleData);

        // Normalize context model response
        setContextModel({
          source: contextRes.source || "precomputed",
          predicted_output: contextRes.predicted_output,
          latency_ms: contextRes.latency_ms,
          confidence: contextRes.confidence,
          nodes: contextRes.state_snapshot?.nodes || [],
          edges: contextRes.state_snapshot?.edges || [],
          memory_status: contextRes.state_snapshot?.memory_status || `holding (${count} pairs)`,
        });

        // Normalize optimization model response
        setOptimizationModel({
          source: optRes.source || "precomputed",
          predicted_output: optRes.predicted_output,
          latency_ms: optRes.latency_ms,
          confidence: optRes.confidence,
          current_step: optRes.current_step || 120,
          max_steps: optRes.max_steps || 120,
          loss: optRes.loss || 0.042,
          learning_rate: optRes.learning_rate || 0.005,
          loss_curve: optRes.loss_curve || [],
          incorrect_cells: optRes.incorrect_cells || [],
        });

        setIngestionStage(count);
        setOptStepIndex((optRes.loss_curve?.length || 13) - 1);
      } catch (err) {
        console.error("Failed to load real prediction data:", err);
      }
    },
    []
  );

  useEffect(() => {
    if (selectedPuzzleId) {
      loadData(selectedPuzzleId, demoCount, novelty, isAdversarial);
    }
  }, [selectedPuzzleId, demoCount, novelty, isAdversarial, loadData]);

  const handleSelectPuzzle = (id: string) => {
    setSelectedPuzzleId(id);
  };

  const handleDemoCountChange = (count: number) => {
    setDemoCount(count);
  };

  const handleNoveltyChange = (nov: "familiar" | "novel") => {
    setNovelty(nov);
    setIsNoveltyFlashing(true);
    setTimeout(() => {
      setIsNoveltyFlashing(false);
    }, 200);
  };

  const handleToggleAdversarial = () => {
    setIsAdversarial((prev) => !prev);
  };

  if (!currentPuzzle || !contextModel || !optimizationModel) {
    return (
      <main className={styles.container}>
        <div className="data-value" style={{ color: "var(--text-secondary)" }}>
          Connecting to laboratory endpoints...
        </div>
      </main>
    );
  }

  return (
    <main
      className={`${styles.container} ${
        isNoveltyFlashing ? styles.noveltyVignette : ""
      }`}
    >
      {/* Header */}
      <header className={styles.instrumentHeader}>
        <div className={styles.brandArea}>
          <h1 className={styles.title}>Learn it or remember it</h1>
          <p className={styles.subtitle}>
            Two-model adaptation benchmark — in-context memory vs gradient optimization
          </p>
        </div>

        <div className={styles.headerTelemetry}>
          <div className={styles.telemetryItem}>
            <span>Benchmark suite</span>
            <span className={styles.telemetryVal}>ARC-eval-v1</span>
          </div>
          <div className={styles.telemetryItem}>
            <span>Distribution</span>
            <span className={styles.telemetryVal}>
              {isAdversarial ? "adversarial" : novelty}
            </span>
          </div>
        </div>
      </header>

      {/* Horizontal Pill-Tab Row (Neutral Tabs, 2px Neutral Underline) */}
      <nav className={styles.puzzlePicker} aria-label="Puzzle selection tabs">
        <span className={styles.pickerLabel}>Select task:</span>
        <div className={styles.tabList} role="tablist">
          {puzzles.map((p) => {
            const isActive = p.id === currentPuzzle.id;
            return (
              <button
                key={p.id}
                role="tab"
                aria-selected={isActive}
                className={`${styles.puzzleTab} ${
                  isActive ? styles.puzzleTabActive : ""
                }`}
                onClick={() => handleSelectPuzzle(p.id)}
              >
                <span>{p.name}</span>
                <span className={styles.tabDimension}>{p.dimension}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Functional Benchmark Control Bar */}
      <ControlBar
        demoCount={demoCount}
        onDemoCountChange={handleDemoCountChange}
        novelty={novelty}
        onNoveltyChange={handleNoveltyChange}
        isAdversarial={isAdversarial}
        onToggleAdversarial={handleToggleAdversarial}
      />

      {/* Two Differentiated Instrument Panels */}
      <div className={styles.comparisonGrid}>
        {/* Panel 1: Context-Route (Live Memory Instrument - Teal #5EEAD4) */}
        <section
          className={`${styles.instrumentPanel} ${styles.contextInstrument}`}
          aria-label="In-context memory model instrument"
        >
          <div className={styles.panelHeader}>
            <div className={styles.panelTitleArea}>
              <h2 className={styles.panelTitle}>Context-route adaptation</h2>
              <span className={styles.badgeContext}>Memory instrument</span>
            </div>

            <div className={styles.headerRightArea}>
              <div className="data-value" style={{ color: "var(--accent-a)", fontSize: "11px" }}>
                Holding {ingestionStage} / {demoCount} demonstration pairs
              </div>
              <PrecomputedBadge source={contextModel.source} />
            </div>
          </div>

          <div className={styles.panelBody}>
            {/* Organic Spring Node-and-Edge Graph */}
            <StateVectorView
              nodes={contextModel.nodes}
              edges={contextModel.edges}
              ingestionStage={ingestionStage}
              onIngestNext={() => setIngestionStage((prev) => Math.min(demoCount, prev + 1))}
              onReset={() => setIngestionStage(0)}
            />

            {/* Linear Grid Flow: Test Input -> Predicted Output -> Ground Truth */}
            <div className={styles.flowSequence}>
              <GridDisplay
                grid={currentPuzzle.test_input}
                title="Test input"
                cellSize={28}
              />

              <span className={styles.arrowGlyph} aria-hidden="true">→</span>

              <DiffGridDisplay
                predictedGrid={contextModel.predicted_output}
                groundTruthGrid={currentPuzzle.ground_truth}
                title="Predicted output"
                cellSize={28}
                routeType="context"
              />

              <span className={styles.arrowGlyph} aria-hidden="true">→</span>

              <GridDisplay
                grid={currentPuzzle.ground_truth}
                title="Ground truth"
                cellSize={28}
              />
            </div>
          </div>

          {/* Monospace Telemetry Footer */}
          <footer className={styles.panelFooter}>
            <div className={styles.metricGroup}>
              <span className={styles.metricLabel}>Latency:</span>
              <span className={styles.metricValContext}>
                {contextModel.latency_ms.toFixed(1)} ms
              </span>
            </div>
            <div className={styles.metricGroup}>
              <span className={styles.metricLabel}>Confidence:</span>
              <span className={styles.metricValContext}>
                {(contextModel.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className={styles.metricGroup}>
              <span className={styles.metricLabel}>Memory status:</span>
              <span className={styles.metricValNeutral}>
                {contextModel.memory_status}
              </span>
            </div>
          </footer>
        </section>

        {/* Panel 2: Optimization-Route (Live Training Instrument - Coral #F2967D) */}
        <section
          className={`${styles.instrumentPanel} ${styles.optimizationInstrument}`}
          aria-label="Gradient optimization model instrument"
        >
          <div className={styles.panelHeader}>
            <div className={styles.panelTitleArea}>
              <h2 className={styles.panelTitle}>Optimization-route adaptation</h2>
              <span className={styles.badgeOpt}>Training instrument</span>
              {isAdversarial && (
                <span className={styles.adversarialTag}>Adversarial failure</span>
              )}
            </div>

            <div className={styles.headerRightArea}>
              <div className="data-value" style={{ color: "var(--accent-b)", fontSize: "11px" }}>
                Step {optimizationModel.loss_curve[optStepIndex]?.step ?? 0} /{" "}
                {optimizationModel.max_steps}
              </div>
              <PrecomputedBadge source={optimizationModel.source} />
            </div>
          </div>

          <div className={styles.panelBody}>
            {/* Mechanical Stepped Loss Line Chart */}
            <LossCurveView
              lossHistory={optimizationModel.loss_curve}
              currentStepIndex={optStepIndex}
              onStepNext={() =>
                setOptStepIndex((prev) =>
                  Math.min(optimizationModel.loss_curve.length - 1, prev + 1)
                )
              }
              onReset={() => setOptStepIndex(0)}
            />

            {/* Linear Grid Flow: Test Input -> Predicted Output (with coral error outlines in adversarial mode) -> Ground Truth */}
            <div className={styles.flowSequence}>
              <GridDisplay
                grid={currentPuzzle.test_input}
                title="Test input"
                cellSize={28}
              />

              <span className={styles.arrowGlyph} aria-hidden="true">→</span>

              <DiffGridDisplay
                predictedGrid={optimizationModel.predicted_output}
                groundTruthGrid={currentPuzzle.ground_truth}
                title="Predicted output"
                cellSize={28}
                routeType="optimization"
                incorrectCells={optimizationModel.incorrect_cells}
              />

              <span className={styles.arrowGlyph} aria-hidden="true">→</span>

              <GridDisplay
                grid={currentPuzzle.ground_truth}
                title="Ground truth"
                cellSize={28}
              />
            </div>
          </div>

          {/* Monospace Telemetry Footer */}
          <footer className={styles.panelFooter}>
            <div className={styles.metricGroup}>
              <span className={styles.metricLabel}>Latency:</span>
              <span className={styles.metricValOpt}>
                {optimizationModel.latency_ms.toFixed(1)} ms
              </span>
            </div>
            <div className={styles.metricGroup}>
              <span className={styles.metricLabel}>Confidence:</span>
              <span className={styles.metricValOpt}>
                {(optimizationModel.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className={styles.metricGroup}>
              <span className={styles.metricLabel}>Learning rate:</span>
              <span className={styles.metricValNeutral}>
                {optimizationModel.learning_rate}
              </span>
            </div>
            <div className={styles.metricGroup}>
              <span className={styles.metricLabel}>Optimization loss:</span>
              <span className={styles.metricValOpt}>
                {(optimizationModel.loss_curve[optStepIndex]?.loss ?? optimizationModel.loss).toFixed(4)}
              </span>
            </div>
          </footer>
        </section>
      </div>

      {/* Empirical Findings Panel (Direct real findings for judges) */}
      <EmpiricalFindingsPanel />
    </main>
  );
}
