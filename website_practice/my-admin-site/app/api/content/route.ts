import { createClient } from "@/lib/supabase/server";

export async function GET(request: Request) {
  const page = new URL(request.url).searchParams.get("page");
  if (page !== "home" && page !== "services") {
    return Response.json({ success: false, error: "Invalid page." }, { status: 400 });
  }

  const supabase = await createClient();
  const { data, error } = await supabase
    .from("site_content")
    .select("id, page_slug, section_key, title, body, cta_label, cta_href, sort_order")
    .eq("page_slug", page)
    .eq("status", "published")
    .order("sort_order");

  if (error) return Response.json({ success: false, error: error.message }, { status: 500 });
  return Response.json({ success: true, content: data ?? [] });
}
