import { NextRequest, NextResponse } from "next/server";
import { MOCK_PUZZLES } from "@/lib/mockApi";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const puzzle = MOCK_PUZZLES.find((p) => p.id === id) || MOCK_PUZZLES[0];
  return NextResponse.json(puzzle);
}
