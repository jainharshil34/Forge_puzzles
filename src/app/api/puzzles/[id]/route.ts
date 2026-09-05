import { NextRequest, NextResponse } from "next/server";
import { MOCK_PUZZLES } from "@/lib/mockApi";
import { transformBackendPuzzle } from "../route";

const PYTHON_BACKEND_URL =
  process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  // Attempt to fetch from Python backend across test and novelty splits
  try {
    const splits = ["test", "novelty", "validation", "train"];
    for (const split of splits) {
      const backendRes = await fetch(
        `${PYTHON_BACKEND_URL}/api/puzzles?split=${split}&limit=100`,
        { cache: "no-store" }
      );
      if (backendRes.ok) {
        const data = await backendRes.json();
        const found = (data.puzzles || []).find((p: any) => p.id === id);
        if (found) {
          return NextResponse.json(transformBackendPuzzle(found));
        }
      }
    }
  } catch (err) {
    console.warn(`Could not reach Python backend for puzzle ${id}, falling back:`, err);
  }

  // Fallback to MOCK_PUZZLES
  const fallback = MOCK_PUZZLES.find((p) => p.id === id) || MOCK_PUZZLES[0];
  return NextResponse.json({
    ...fallback,
    dimension: "5×5",
  });
}
