import { createClient } from "@/lib/supabase/server";

export async function GET() {
  const supabase = await createClient();
  const { data, error } = await supabase
    .from("articles")
    .select("id, title, excerpt, content, author_name, created_at")
    .eq("status", "published")
    .order("created_at", { ascending: false });
  if (error) return Response.json({ success: false, error: error.message }, { status: 500 });
  return Response.json({ success: true, articles: data });
}
