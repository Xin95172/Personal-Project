import { requireAdmin } from "@/lib/admin";

function error(message: string, status: number) {
  return Response.json({ success: false, error: message }, { status });
}

export async function GET() {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const { data, error: queryError } = await access.supabase
    .from("articles")
    .select("id, title, excerpt, content, author_name, status, created_at, updated_at")
    .order("created_at", { ascending: false });
  if (queryError) return error(queryError.message, 500);
  return Response.json({ success: true, articles: data });
}

export async function POST(request: Request) {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const title = typeof body?.title === "string" ? body.title.trim() : "";
  const authorName = typeof body?.authorName === "string" ? body.authorName.trim() : "";
  const excerpt = typeof body?.excerpt === "string" ? body.excerpt.trim() : "";
  const content = typeof body?.content === "string" ? body.content.trim() : "";
  const status = body?.status === "published" ? "published" : "draft";
  if (!title || !authorName || !content) return error("標題、作者與文章內容為必填", 400);
  const { data, error: insertError } = await access.supabase
    .from("articles")
    .insert({ title, author_name: authorName, excerpt, content, status })
    .select("id, title, excerpt, content, author_name, status, created_at, updated_at")
    .single();
  if (insertError) return error(insertError.message, 500);
  return Response.json({ success: true, article: data }, { status: 201 });
}

export async function PATCH(request: Request) {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const id = typeof body?.id === "string" ? body.id : "";
  const status = body?.status === "published" ? "published" : body?.status === "draft" ? "draft" : null;
  if (!id || !status) return error("資料格式錯誤", 400);
  const { error: updateError } = await access.supabase.from("articles").update({ status }).eq("id", id);
  if (updateError) return error(updateError.message, 500);
  return Response.json({ success: true });
}

export async function DELETE(request: Request) {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const id = typeof body?.id === "string" ? body.id : "";
  if (!id) return error("缺少文章識別碼", 400);
  const { error: deleteError } = await access.supabase.from("articles").delete().eq("id", id);
  if (deleteError) return error(deleteError.message, 500);
  return Response.json({ success: true });
}
