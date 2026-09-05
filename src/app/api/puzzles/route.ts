import { NextRequest, NextResponse } from "next/server";
import { MOCK_PUZZLES } from "@/lib/mockApi";

const PYTHON_BACKEND_URL =
  process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000";

function formatRuleName(ruleType: string, ruleParams: any): string {
  if (ruleType === "translate") {
    const dir = ruleParams?.direction || "right";
    const mag = ruleParams?.magnitude ?? 1;
    return `Translate ${dir} by ${mag}`;
  }
  if (ruleType === "mirror") {
    const axis = ruleParams?.axis || "horizontal";
    return `Mirror ${axis}`;
  }
  if (ruleType === "recolor") {
    return "Recolor permutation";
  }
  if (!ruleType) return "ARC reasoning task";
  return `${ruleType.charAt(0).toUpperCase() + ruleType.slice(1)}`;
}

export function transformBackendPuzzle(p: any) {
  return {
    id: p.id,
    name: formatRuleName(p.rule_type, p.rule_params),
    rule_type: p.rule_type,
    rule_params: p.rule_params,
    dimension: "5×5",
    demo_pairs: p.demo_pairs || [],
    test_pair: p.test_pair || { input: [], output: [] },
    test_input: p.test_pair?.input || p.test_input || [],
    ground_truth: p.test_pair?.output || p.ground_truth || [],
  };
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const split = searchParams.get("split") || "test";
  const limit = searchParams.get("limit") || "20";

  try {
    const backendRes = await fetch(
      `${PYTHON_BACKEND_URL}/api/puzzles?split=${encodeURIComponent(
        split
      )}&limit=${encodeURIComponent(limit)}`,
      { cache: "no-store" }
    );

    if (backendRes.ok) {
      const data = await backendRes.json();
      const rawPuzzles = data.puzzles || [];
      if (Array.isArray(rawPuzzles) && rawPuzzles.length > 0) {
        const transformed = rawPuzzles.map(transformBackendPuzzle);
        return NextResponse.json(transformed);
      }
    }
  } catch (err) {
    console.warn("Could not reach Python backend for puzzles list, falling back to mock:", err);
  }

  // Fallback to MOCK_PUZZLES if backend is offline
  return NextResponse.json(
    MOCK_PUZZLES.map((p) => ({
      ...p,
      dimension: "5×5",
    }))
  );
}
