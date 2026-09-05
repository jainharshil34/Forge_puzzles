import { NextResponse } from "next/server";
import { MOCK_PUZZLES } from "@/lib/mockApi";

export async function GET() {
  const summary = MOCK_PUZZLES.map((p) => ({
    id: p.id,
    name: p.name,
    dimension: p.dimension,
    demo_count: p.demo_pairs.length,
    test_input: p.test_input,
    ground_truth: p.ground_truth,
  }));
  return NextResponse.json(summary);
}
