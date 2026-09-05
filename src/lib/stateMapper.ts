/**
 * stateMapper.ts - Dimensionality-reduction projection for recurrent GRU state vectors.
 *
 * NOTE: This is a dimensionality-reduction visualization of the real high-dimensional
 * (128-d) state vector, projecting contiguous latent feature partitions onto an intuitive
 * 10-node topological associative graph, rather than a literal rendering of all 128 raw dimensions.
 */

export interface GraphNode {
  id: string;
  label: string;
  x: number;
  y: number;
  layer: number;
  activation?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  weights: number[]; // weights progression across [baseline, demo1, demo2, demo3, demo4, demo5]
}

export interface PerDemoState {
  demo_index: number;
  state_vector: number[];
  state_norm?: number;
  state_dimension?: number;
}

export const STANDARD_GRAPH_NODES: GraphNode[] = [
  // Layer 0: Context input feature nodes (x: 36)
  { id: "n0", label: "C0", x: 36, y: 35, layer: 0 },
  { id: "n1", label: "C1", x: 36, y: 75, layer: 0 },
  { id: "n2", label: "C2", x: 36, y: 115, layer: 0 },
  { id: "n3", label: "C3", x: 36, y: 155, layer: 0 },

  // Layer 1: Associative latent memory cluster (x: 135)
  { id: "n4", label: "M0", x: 135, y: 30, layer: 1 },
  { id: "n5", label: "M1", x: 135, y: 65, layer: 1 },
  { id: "n6", label: "M2", x: 135, y: 100, layer: 1 },
  { id: "n7", label: "M3", x: 135, y: 135, layer: 1 },
  { id: "n8", label: "M4", x: 135, y: 170, layer: 1 },

  // Layer 2: Output binding node (x: 230)
  { id: "n9", label: "R", x: 230, y: 95, layer: 2 },
];

export const TOPOLOGICAL_EDGES: Array<{ id: string; source: string; target: string }> = [
  { id: "e0", source: "n0", target: "n4" },
  { id: "e1", source: "n0", target: "n5" },
  { id: "e2", source: "n1", target: "n5" },
  { id: "e3", source: "n1", target: "n6" },
  { id: "e4", source: "n2", target: "n6" },
  { id: "e5", source: "n2", target: "n7" },
  { id: "e6", source: "n3", target: "n7" },
  { id: "e7", source: "n3", target: "n8" },
  { id: "e8", source: "n4", target: "n5" },
  { id: "e9", source: "n5", target: "n6" },
  { id: "e10", source: "n6", target: "n7" },
  { id: "e11", source: "n7", target: "n8" },
  { id: "e12", source: "n4", target: "n9" },
  { id: "e13", source: "n5", target: "n9" },
  { id: "e14", source: "n6", target: "n9" },
  { id: "e15", source: "n7", target: "n9" },
  { id: "e16", source: "n8", target: "n9" },
  { id: "e17", source: "n1", target: "n7" },
];

/**
 * Projects a raw state vector of length D (e.g. 128) onto 10 node activations
 * by computing the mean absolute value of each contiguous segment.
 */
export function projectStateVectorToNodeActivations(
  stateVector: number[]
): number[] {
  if (!Array.isArray(stateVector) || stateVector.length === 0) {
    return new Array(10).fill(0.1);
  }

  const numBuckets = 10;
  const bucketSize = Math.floor(stateVector.length / numBuckets);
  const activations: number[] = [];

  for (let i = 0; i < numBuckets; i++) {
    const start = i * bucketSize;
    const end = i === numBuckets - 1 ? stateVector.length : start + bucketSize;
    const segment = stateVector.slice(start, end);

    if (segment.length === 0) {
      activations.push(0.1);
      continue;
    }

    const sumAbs = segment.reduce((acc, val) => acc + Math.abs(val), 0);
    const meanAbs = sumAbs / segment.length;

    // Scale to a clean normalized activation range [0.08, 1.0]
    const scaled = Math.min(1.0, Math.max(0.08, meanAbs * 1.8));
    activations.push(parseFloat(scaled.toFixed(3)));
  }

  return activations;
}

/**
 * Generates an array of graph progression edge weights from the backend's real per_demo_states.
 * Step 0 represents baseline, steps 1..N represent demo step states.
 */
export function generateGraphProgression(perDemoStates: PerDemoState[]): {
  nodes: GraphNode[];
  edges: GraphEdge[];
} {
  const nodeIndexMap: Record<string, number> = {
    n0: 0,
    n1: 1,
    n2: 2,
    n3: 3,
    n4: 4,
    n5: 5,
    n6: 6,
    n7: 7,
    n8: 8,
    n9: 9,
  };

  // Step 0 baseline activations (minimal background potential)
  const stepActivations: number[][] = [new Array(10).fill(0.12)];

  // Compute activations for each demo step from the real state_vector
  (perDemoStates || []).forEach((state) => {
    const act = projectStateVectorToNodeActivations(state.state_vector || []);
    stepActivations.push(act);
  });

  // Ensure at least 6 steps [0..5] for smooth slider tracking
  while (stepActivations.length <= 5) {
    const last = stepActivations[stepActivations.length - 1];
    stepActivations.push([...last]);
  }

  // Compute edge weights across all steps
  const edges: GraphEdge[] = TOPOLOGICAL_EDGES.map((edge) => {
    const uIdx = nodeIndexMap[edge.source] ?? 0;
    const vIdx = nodeIndexMap[edge.target] ?? 0;

    const weights = stepActivations.map((acts) => {
      const uAct = acts[uIdx] ?? 0.1;
      const vAct = acts[vIdx] ?? 0.1;
      const weight = (uAct + vAct) / 2.0;
      return parseFloat(Math.min(1.0, Math.max(0.05, weight)).toFixed(3));
    });

    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      weights,
    };
  });

  return {
    nodes: STANDARD_GRAPH_NODES,
    edges,
  };
}
