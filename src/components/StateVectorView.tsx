"use client";

import React from "react";
import { motion, useReducedMotion } from "framer-motion";
import styles from "./StateVectorView.module.css";

export interface GraphNode {
  id: string;
  label: string;
  x: number;
  y: number;
  layer: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  weights?: number[] | number;
}

export interface StateVectorViewProps {
  nodes?: GraphNode[];
  edges?: GraphEdge[];
  ingestionStage: number; // 0: baseline, 1: demo 1 ingested, etc.
  onIngestNext?: () => void;
  onReset?: () => void;
  className?: string;
}

const DEFAULT_NODES: GraphNode[] = [
  { id: "n0", label: "C0", x: 36, y: 35, layer: 0 },
  { id: "n1", label: "C1", x: 36, y: 75, layer: 0 },
  { id: "n2", label: "C2", x: 36, y: 115, layer: 0 },
  { id: "n3", label: "C3", x: 36, y: 155, layer: 0 },
  { id: "n4", label: "M0", x: 135, y: 30, layer: 1 },
  { id: "n5", label: "M1", x: 135, y: 65, layer: 1 },
  { id: "n6", label: "M2", x: 135, y: 100, layer: 1 },
  { id: "n7", label: "M3", x: 135, y: 135, layer: 1 },
  { id: "n8", label: "M4", x: 135, y: 170, layer: 1 },
  { id: "n9", label: "R", x: 230, y: 95, layer: 2 },
];

export const StateVectorView: React.FC<StateVectorViewProps> = ({
  nodes = DEFAULT_NODES,
  edges = [],
  ingestionStage,
  onIngestNext,
  onReset,
  className = "",
}) => {
  const prefersReducedMotion = useReducedMotion();

  // Create node lookup map
  const nodeMap = React.useMemo(() => {
    const map = new Map<string, GraphNode>();
    (nodes || []).forEach((n) => map.set(n.id, n));
    return map;
  }, [nodes]);

  const springTransition = (index: number) => {
    if (prefersReducedMotion) {
      return { duration: 0 };
    }
    return {
      type: "spring" as const,
      stiffness: 300,
      damping: 15,
      delay: index * 0.015, // ~15ms stagger per edge
    };
  };

  const getEdgeProps = (edge: GraphEdge, index: number) => {
    let weight = 0.2;
    if (Array.isArray(edge.weights)) {
      weight = edge.weights[Math.min(ingestionStage, edge.weights.length - 1)] ?? 0.2;
    } else if (typeof edge.weights === "number") {
      weight = edge.weights;
    }

    const strokeWidth = 0.8 + weight * 3.4;
    const strokeOpacity = 0.15 + weight * 0.8;
    const strokeColor = `rgba(94, 234, 212, ${strokeOpacity})`;

    return {
      strokeWidth,
      stroke: strokeColor,
      transition: springTransition(index),
    };
  };

  return (
    <div className={`${styles.container} ${className}`}>
      <div className={styles.headerRow}>
        <div className={styles.titleArea}>
          <h3 className={styles.title}>Synaptic memory state graph</h3>
          <span className={styles.statusTag}>Live memory</span>
        </div>
        <span className={styles.graphStageBadge}>
          {ingestionStage === 0
            ? "Baseline (0 pairs)"
            : `Holding ${ingestionStage} pairs`}
        </span>
      </div>

      <div className={styles.svgWrapper}>
        <svg
          viewBox="0 0 270 200"
          className={styles.graphSvg}
          aria-label="Node and edge state vector memory graph"
        >
          {/* Edge Connection Layer */}
          <g className="edges">
            {(edges || []).map((edge, idx) => {
              const src = nodeMap.get(edge.source);
              const tgt = nodeMap.get(edge.target);
              if (!src || !tgt) return null;

              const edgeAnim = getEdgeProps(edge, idx);

              return (
                <motion.line
                  key={`${edge.id}-${ingestionStage}`}
                  x1={src.x}
                  y1={src.y}
                  x2={tgt.x}
                  y2={tgt.y}
                  initial={
                    prefersReducedMotion
                      ? undefined
                      : { strokeWidth: 0.8, stroke: "rgba(94, 234, 212, 0.15)" }
                  }
                  animate={{
                    strokeWidth: edgeAnim.strokeWidth,
                    stroke: edgeAnim.stroke,
                  }}
                  transition={edgeAnim.transition}
                  strokeLinecap="round"
                />
              );
            })}
          </g>

          {/* Node Layer */}
          <g className="nodes">
            {(nodes || []).map((node, nIdx) => {
              const isActive = ingestionStage > 0;
              const nodeDelay = prefersReducedMotion ? 0 : nIdx * 0.015;

              return (
                <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                  <motion.circle
                    r={node.layer === 2 ? 8 : 6}
                    fill="#171A21"
                    stroke="#2A2E38"
                    strokeWidth={1}
                    animate={
                      prefersReducedMotion
                        ? undefined
                        : {
                            scale: isActive ? [1, 1.28, 1] : 1,
                            stroke: isActive ? "#5EEAD4" : "#2A2E38",
                          }
                    }
                    transition={{
                      type: "spring",
                      stiffness: 300,
                      damping: 15,
                      delay: nodeDelay,
                    }}
                  />
                  <circle
                    r={node.layer === 2 ? 3 : 2}
                    fill={isActive ? "#5EEAD4" : "#8B909C"}
                  />
                  <text y={node.layer === 0 ? -10 : 12} className={styles.nodeLabel}>
                    {node.label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      <p className={styles.caption}>
        <span className={styles.captionHighlight}>Synaptic memory:</span> Ingesting
        demonstration pairs strengthens associative pathways in real time via spring
        dynamics without backpropagation or weight mutations.
      </p>

      <div className={styles.controlsRow}>
        <button
          type="button"
          className={styles.ingestBtn}
          onClick={onIngestNext}
        >
          Ingest demo pair
        </button>
        {ingestionStage > 0 && (
          <button
            type="button"
            className={styles.resetBtn}
            onClick={onReset}
          >
            Reset graph
          </button>
        )}
      </div>
    </div>
  );
};
