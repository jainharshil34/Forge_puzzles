import { NextResponse } from "next/server";

const PYTHON_BACKEND_URL =
  process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000";

export async function GET() {
  try {
    const res = await fetch(`${PYTHON_BACKEND_URL}/api/results`, {
      cache: "no-store",
    });

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json({
        source: "live_backend_results",
        sweep: data.sweep || [],
        forgetting: data.forgetting || null,
        state_capacity: data.state_capacity || [],
        generalization: data.generalization || null,
      });
    }
  } catch (err) {
    console.warn("Could not reach Python backend for results, using fallback:", err);
  }

  // Fallback empirical dataset mirroring results/*.json
  return NextResponse.json({
    source: "precomputed_fallback",
    sweep: [
      { demo_count: 1, exact_match: 0.02, cell_accuracy: 0.7752, model: "context", rule_type: "translate", latency_ms: 0.53 },
      { demo_count: 2, exact_match: 0.54, cell_accuracy: 0.9640, model: "context", rule_type: "translate", latency_ms: 0.92 },
      { demo_count: 3, exact_match: 0.92, cell_accuracy: 0.9968, model: "context", rule_type: "translate", latency_ms: 1.41 },
      { demo_count: 4, exact_match: 0.96, cell_accuracy: 0.9976, model: "context", rule_type: "translate", latency_ms: 1.95 },
      { demo_count: 5, exact_match: 0.98, cell_accuracy: 0.9992, model: "context", rule_type: "translate", latency_ms: 2.45 },
      { gradient_steps: 0, exact_match: 0.0, cell_accuracy: 0.4888, model: "optimization", rule_type: "translate", latency_ms: 0.79 },
      { gradient_steps: 1, exact_match: 0.0, cell_accuracy: 0.4904, model: "optimization", rule_type: "translate", latency_ms: 30.31 },
      { gradient_steps: 3, exact_match: 0.0, cell_accuracy: 0.4904, model: "optimization", rule_type: "translate", latency_ms: 55.40 },
      { gradient_steps: 5, exact_match: 0.0, cell_accuracy: 0.4912, model: "optimization", rule_type: "translate", latency_ms: 82.15 },
      { gradient_steps: 10, exact_match: 0.0, cell_accuracy: 0.4944, model: "optimization", rule_type: "translate", latency_ms: 145.20 },
    ],
    forgetting: {
      summary: {
        optimization_forgetting: 0.0,
        context_forgetting: 0.0,
      },
    },
    state_capacity: [
      { state_size: 32, test_exact_match: 0.27, test_cell_accuracy: 0.9298 },
      { state_size: 64, test_exact_match: 0.35, test_cell_accuracy: 0.9512 },
      { state_size: 128, test_exact_match: 0.315, test_cell_accuracy: 0.9440 },
      { state_size: 256, test_exact_match: 0.32, test_cell_accuracy: 0.9428 },
      { state_size: 512, test_exact_match: 0.17, test_cell_accuracy: 0.9256 },
    ],
  });
}
