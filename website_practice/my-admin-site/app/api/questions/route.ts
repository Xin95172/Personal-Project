import { createClient } from "@/lib/supabase/server";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const pseudonym = typeof body?.pseudonym === "string" ? body.pseudonym.trim().slice(0, 80) : "匿名";
  const question = typeof body?.question === "string" ? body.question.trim().slice(0, 2000) : "";
  if (!question) return Response.json({ success: false, error: "請輸入問題內容" }, { status: 400 });
  const supabase = await createClient();
  const { error } = await supabase.from("question_submissions").insert({ pseudonym: pseudonym || "匿名", question });
  if (error) return Response.json({ success: false, error: error.message }, { status: 500 });
  return Response.json({ success: true }, { status: 201 });
}
