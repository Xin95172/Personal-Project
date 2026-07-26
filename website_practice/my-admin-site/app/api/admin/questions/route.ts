import { requireAdmin } from "@/lib/admin";

function error(message: string, status: number) { return Response.json({ success: false, error: message }, { status }); }

export async function GET() {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const { data, error: queryError } = await access.supabase.from("question_submissions").select("id, pseudonym, question, status, answer, created_at, answered_at").order("created_at", { ascending: false });
  if (queryError) return error(queryError.message, 500);
  return Response.json({ success: true, questions: data });
}

export async function PATCH(request: Request) {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const id = typeof body?.id === "string" ? body.id : "";
  const status = ["pending", "selected", "answered", "rejected"].includes(String(body?.status)) ? String(body?.status) : "";
  const answer = typeof body?.answer === "string" ? body.answer.trim() : null;
  if (!id || !status) return error("資料格式錯誤", 400);
  const values = { status, answer, answered_at: status === "answered" ? new Date().toISOString() : null };
  const { error: updateError } = await access.supabase.from("question_submissions").update(values).eq("id", id);
  if (updateError) return error(updateError.message, 500);
  return Response.json({ success: true });
}
