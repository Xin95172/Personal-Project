"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { PenLine } from "lucide-react";
import SiteHeader from "@/components/site-header";

type Article = { id: string; title: string; excerpt: string; content: string; author_name: string };

export default function ColumnsPage() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const response = await fetch("/api/articles", { cache: "no-store" });
      const data = await response.json();
      if (response.ok) setArticles(data.articles ?? []);
      setLoading(false);
    }
    void load();
  }, []);

  return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader/><main className="mx-auto max-w-7xl px-5 py-16 lg:px-8"><div className="flex flex-col justify-between gap-6 md:flex-row md:items-end"><div><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">COLUMNS</p><h1 className="mt-4 text-4xl font-bold text-[#173f3b]">專欄作者的品牌觀點</h1><p className="mt-5 max-w-2xl leading-8 text-stone-600">授權作者發表商標、品牌與權利規劃內容的園地。</p></div><Link href="/columns/submit" className="inline-flex w-fit items-center gap-2 rounded-full bg-[#173f3b] px-5 py-3 text-sm font-bold text-white"><PenLine size={16}/> 作者投稿</Link></div><section className="mt-12 grid gap-6 md:grid-cols-3">{loading && <p className="text-stone-500">文章載入中…</p>}{!loading && articles.map((article, index) => <article key={article.id} className="flex min-h-80 flex-col rounded-2xl border border-stone-200 bg-white p-7"><p className="text-sm font-bold text-[#a57a2c]">COLUMN · {String(index + 1).padStart(2, "0")}</p><h2 className="mt-10 text-xl font-bold leading-8 text-[#173f3b]">{article.title}</h2><p className="mt-4 leading-7 text-stone-600">{article.excerpt || article.content.slice(0, 110)}</p><p className="mt-auto pt-8 text-sm font-medium text-stone-500">作者／{article.author_name}</p></article>)}{!loading && articles.length === 0 && <div className="col-span-full rounded-2xl border border-dashed border-stone-300 bg-white px-6 py-16 text-center text-stone-500">目前尚無已發布文章。</div>}</section></main></div>;
}
