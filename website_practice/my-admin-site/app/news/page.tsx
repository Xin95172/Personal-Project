import Link from "next/link";
import { ArrowRight } from "lucide-react";
import SiteHeader from "@/components/site-header";
import { createClient } from "@/lib/supabase/server";
import { connection } from "next/server";

export default async function NewsPage() {
  await connection();
  const supabase = await createClient();
  const { data: articles } = await supabase.from("articles").select("id, title, excerpt, content, author_name, created_at").eq("status", "published").order("created_at", { ascending: false });
  return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader/><main className="mx-auto max-w-6xl px-5 py-16 lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">TRADEMARK NOTES</p><h1 className="mt-4 text-4xl font-bold text-[#173f3b]">商標觀點與最新文章</h1><p className="mt-4 max-w-2xl leading-7 text-stone-600">由網站管理員發布的文章會即時顯示在這裡。</p><div className="mt-10 grid gap-6 md:grid-cols-3">{articles?.map((article, i) => <article className="rounded-2xl border border-stone-200 bg-white p-7" key={article.id}><p className="text-sm font-bold text-[#a57a2c]">INSIGHT · {String(i + 1).padStart(2, "0")}</p><h2 className="mt-10 text-xl font-bold leading-8 text-[#173f3b]">{article.title}</h2><p className="mt-3 text-sm text-stone-500">{article.author_name}</p><p className="mt-4 whitespace-pre-line leading-7 text-stone-600">{article.excerpt || article.content.slice(0, 140)}</p><Link href="/contact" className="mt-8 inline-flex items-center gap-2 text-sm font-bold text-[#173f3b]">洽詢服務<ArrowRight size={16}/></Link></article>)}{!articles?.length && <p className="col-span-full rounded-2xl border border-dashed border-stone-300 p-10 text-center text-stone-500">目前尚無已發布文章。</p>}</div></main></div>;
}
