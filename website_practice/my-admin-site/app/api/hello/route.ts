import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    success: true,
    message: "後端 API 正常運作",
    serverTime: new Date().toISOString(),
  });
}