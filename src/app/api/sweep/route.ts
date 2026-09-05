import { NextResponse } from "next/server";

export async function GET() {
  const sweepData = {
    source: "precomputed",
    benchmark_name: "ARC-eval-v1-sweep",
    demo_counts: [1, 2, 3, 4, 5],
    metrics: {
      context: {
        latency_trend_ms: [3.1, 3.8, 4.3, 4.9, 5.4],
        accuracy_familiar: [0.92, 0.96, 0.98, 0.99, 1.0],
        accuracy_novel: [0.85, 0.91, 0.94, 0.96, 0.97],
      },
      optimization: {
        latency_trend_ms: [180, 240, 310, 390, 480],
        accuracy_familiar: [0.78, 0.88, 0.91, 0.93, 0.94],
        accuracy_novel: [0.62, 0.74, 0.81, 0.84, 0.86],
      },
    },
  };
  return NextResponse.json(sweepData);
}
