import { NextRequest, NextResponse } from "next/server";
import { MOCK_PUZZLES } from "@/lib/mockApi";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const puzzleId = searchParams.get("puzzle_id") || "arc-sym-01";
  const modelType = searchParams.get("model_type") || "context";
  const demoCount = parseInt(searchParams.get("demo_count") || "2", 10);
  const novelty = searchParams.get("novelty") || "familiar";
  const isAdversarial = searchParams.get("adversarial") === "true";

  const puzzle = MOCK_PUZZLES.find((p) => p.id === puzzleId) || MOCK_PUZZLES[0];
  const isNovel = novelty === "novel";

  // Dynamic source tag: Live when novelty/adversarial active, precomputed for standard cache checkpoints
  const source = isNovel || isAdversarial || demoCount > 3 ? "live" : "precomputed";

  if (modelType === "context") {
    let confidence = 0.94 + demoCount * 0.012;
    if (isNovel) confidence -= 0.05;
    if (isAdversarial) confidence -= 0.08;

    const latency = parseFloat((2.6 + demoCount * 0.55 + (isNovel ? 0.8 : 0)).toFixed(1));
    const predicted = puzzle.ground_truth.map((row) => [...row]);

    // Build state snapshot
    const edges = puzzle.base_edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      weights: [
        e.base,
        Math.min(1.0, e.base + e.perDemo * 1),
        Math.min(1.0, e.base + e.perDemo * 2),
        Math.min(1.0, e.base + e.perDemo * 3),
        Math.min(1.0, e.base + e.perDemo * 4),
        Math.min(1.0, e.base + e.perDemo * 5),
      ],
    }));

    return NextResponse.json({
      source,
      puzzle_id: puzzle.id,
      model_type: "context",
      predicted_output: predicted,
      latency_ms: latency,
      confidence: parseFloat(confidence.toFixed(3)),
      state_snapshot: {
        nodes: [
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
        ],
        edges,
        memory_status: `holding (${demoCount} pairs in active cache)`,
      },
    });
  }

  // Optimization model prediction
  const maxSteps = isNovel ? 180 : 120;
  const stepCount = 12;
  const initialLoss = isAdversarial ? 1.45 : isNovel ? 1.15 : 0.895;
  const minLoss = isAdversarial ? 0.385 : isNovel ? 0.112 : 0.042;

  const lossHistory = [];
  for (let i = 0; i <= stepCount; i++) {
    const ratio = i / stepCount;
    const stepVal = Math.round(ratio * maxSteps);
    const lossVal = minLoss + (initialLoss - minLoss) * Math.exp(-ratio * 3.8);
    lossHistory.push({
      step: stepVal,
      loss: parseFloat(lossVal.toFixed(4)),
    });
  }

  let optConfidence = 0.82 + demoCount * 0.015;
  if (isNovel) optConfidence -= 0.12;
  if (isAdversarial) optConfidence -= 0.38;

  const optLatency = parseFloat((160 + (isNovel ? 90 : 0) + (isAdversarial ? 120 : 0)).toFixed(1));

  const optPredicted = puzzle.ground_truth.map((row) =>
    row.map((val) => (val > 0 ? 2 : 0))
  );

  let incorrectCells: [number, number][] = [];

  if (isAdversarial) {
    if (puzzle.id === "arc-sym-01") {
      incorrectCells = [
        [0, 1],
        [0, 3],
        [2, 2],
        [4, 1],
        [4, 3],
      ];
    } else if (puzzle.id === "arc-bnd-02") {
      incorrectCells = [
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
      ];
    } else {
      incorrectCells = [
        [1, 1],
        [1, 3],
        [2, 2],
        [3, 1],
      ];
    }

    incorrectCells.forEach(([r, c]) => {
      if (optPredicted[r] && optPredicted[r][c] !== undefined) {
        optPredicted[r][c] = optPredicted[r][c] === 2 ? 0 : 2;
      }
    });
  } else if (isNovel) {
    incorrectCells = [[0, 0]];
    optPredicted[0][0] = 2;
  }

  return NextResponse.json({
    source,
    puzzle_id: puzzle.id,
    model_type: "optimization",
    predicted_output: optPredicted,
    latency_ms: optLatency,
    confidence: parseFloat(Math.max(0.25, optConfidence).toFixed(3)),
    current_step: maxSteps,
    max_steps: maxSteps,
    loss: lossHistory[lossHistory.length - 1].loss,
    learning_rate: isNovel ? 0.003 : 0.005,
    loss_curve: lossHistory,
    incorrect_cells: incorrectCells,
  });
}
