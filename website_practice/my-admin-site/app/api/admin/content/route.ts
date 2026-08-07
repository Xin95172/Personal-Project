import { requireAdmin } from "@/lib/admin";

function error(message: string, status: number) {
  return Response.json({ success: false, error: message }, { status });
}

const validPages = ["home", "services"];

export async function GET() {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const { data, error: queryError } = await access.supabase
    .from("site_content")
    .select("id, page_slug, section_key, title, body, cta_label, cta_href, sort_order, status")
    .order("page_slug")
    .order("sort_order");
  if (queryError) return error(queryError.message, 500);
  return Response.json({ success: true, content: data ?? [] });
}

export async function POST(request: Request) {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const pageSlug = typeof body?.pageSlug === "string" ? body.pageSlug : "";
  const sectionKey = typeof body?.sectionKey === "string" ? body.sectionKey.trim().toLowerCase().replace(/[^a-z0-9_-]/g, "-") : "";
  const title = typeof body?.title === "string" ? body.title.trim() : "";
  const content = typeof body?.body === "string" ? body.body.trim() : "";
  if (!validPages.includes(pageSlug) || !sectionKey || !title) return error("Page, section key, and title are required.", 400);
  const { error: insertError } = await access.supabase.from("site_content").insert({
    page_slug: pageSlug, section_key: sectionKey, title, body: content,
    cta_label: typeof body?.ctaLabel === "string" ? body.ctaLabel.trim() : "",
    cta_href: typeof body?.ctaHref === "string" ? body.ctaHref.trim() : "",
    sort_order: Number.isFinite(Number(body?.sortOrder)) ? Number(body?.sortOrder) : 0,
    status: body?.status === "published" ? "published" : "draft",
  });
  if (insertError) return error(insertError.message, 500);
  return Response.json({ success: true }, { status: 201 });
}

export async function PATCH(request: Request) {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const id = typeof body?.id === "string" ? body.id : "";
  const title = typeof body?.title === "string" ? body.title.trim() : "";
  if (!id || !title) return error("Content item and title are required.", 400);
  const { error: updateError } = await access.supabase.from("site_content").update({
    title, body: typeof body?.body === "string" ? body.body.trim() : "",
    cta_label: typeof body?.ctaLabel === "string" ? body.ctaLabel.trim() : "",
    cta_href: typeof body?.ctaHref === "string" ? body.ctaHref.trim() : "",
    sort_order: Number.isFinite(Number(body?.sortOrder)) ? Number(body?.sortOrder) : 0,
    status: body?.status === "published" ? "published" : "draft",
    updated_at: new Date().toISOString(),
  }).eq("id", id);
  if (updateError) return error(updateError.message, 500);
  return Response.json({ success: true });
}

export async function DELETE(request: Request) {
  const access = await requireAdmin();
  if (access.error) return error(access.error, access.status);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  const id = typeof body?.id === "string" ? body.id : "";
  if (!id) return error("Content item is required.", 400);
  const { error: deleteError } = await access.supabase.from("site_content").delete().eq("id", id);
  if (deleteError) return error(deleteError.message, 500);
  return Response.json({ success: true });
}
