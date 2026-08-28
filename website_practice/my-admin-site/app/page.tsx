import Link from "next/link";
import { ArrowRight, FileText, Search, ShieldCheck } from "lucide-react";
import SiteHeader from "@/components/site-header";
import TrademarkSearch from "@/components/trademark-search";
import { createClient } from "@/lib/supabase/server";
import { isSupabaseConfigured } from "@/lib/supabase/config";
import { withTimeout } from "@/lib/with-timeout";

export const dynamic = "force-dynamic";

const fallbackHero = { title: "商標權益，從清楚的第一步開始", body: "提供商標檢索、申請策略與品牌保護建議，協助你在重要決策前掌握資訊。", cta_label: "了解服務", cta_href: "/services" };

export default async function HomePage() {
  const configured = isSupabaseConfigured();
  let data: typeof fallbackHero | null = null;

  if (configured) {
    try {
      const supabase = await createClient();
      const response = await withTimeout(supabase
        .from("site_content")
        .select("title, body, cta_label, cta_href")
        .eq("page_slug", "home")
        .eq("section_key", "hero")
        .eq("status", "published")
        .maybeSingle());

      data = response.data;
    } catch (error) {
      console.error("Unable to load homepage CMS content:", error);
    }
  }
  const hero = data ?? fallbackHero;
  const highlights = [[Search, "商標檢索", "先了解可能的近似商標與申請風險。"], [FileText, "申請規劃", "依品牌定位整理申請方向與類別。"], [ShieldCheck, "品牌保護", "讓商標使用與權利維護更有依據。"]] as const;
  return <div className="min-h-screen bg-[#fbfaf7] text-stone-900"><SiteHeader/><main>{!configured && <div className="bg-amber-50 px-5 py-3 text-center text-sm text-amber-900">目前使用示範內容；設定 Supabase 後即可啟用後台內容管理。</div>}<section className="relative overflow-hidden bg-[#e9e3d5]"><div className="absolute -right-20 top-0 h-full w-[43%] bg-[#d9c39a]/35"/><div className="relative mx-auto grid max-w-7xl gap-14 px-5 py-20 lg:grid-cols-[1.15fr_.85fr] lg:px-8 lg:py-28"><div className="max-w-3xl"><p className="mb-6 text-sm font-bold tracking-[.16em] text-[#9c6c1c]">TRADEMARK COUNSEL</p><h1 className="display-serif text-5xl leading-[1.18] tracking-tight text-[#173f3b] md:text-6xl">{hero.title}</h1><p className="mt-7 max-w-xl whitespace-pre-line text-base leading-8 text-stone-600 md:text-lg">{hero.body}</p><div className="mt-10 flex flex-wrap gap-3"><Link href={hero.cta_href || "/services"} className="inline-flex items-center gap-2 rounded-full bg-[#173f3b] px-6 py-3.5 text-sm font-bold text-white">{hero.cta_label || "了解服務"}<ArrowRight size={17}/></Link><Link href="/contact" className="rounded-full border border-[#173f3b]/30 bg-white/60 px-6 py-3.5 text-sm font-bold text-[#173f3b]">聯絡諮詢</Link></div></div><div className="flex items-center justify-center"><div className="grid aspect-square w-full max-w-[360px] place-items-center rounded-full border-[18px] border-[#f7f0df] bg-[#173f3b]"><div className="text-center text-[#f7f0df]"><p className="display-serif text-7xl">商標</p><p className="mt-3 text-xs font-semibold tracking-[.3em]">BRAND · RIGHTS · TRUST</p></div></div></div></div></section><section className="mx-auto max-w-7xl px-5 py-20 lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">TRADEMARK SEARCH</p><h2 className="mt-3 text-3xl font-bold text-[#173f3b]">查詢商標資料</h2><div className="mt-9 rounded-[2rem] bg-[#d8b66e] p-5 md:p-8"><TrademarkSearch/></div></section><section className="border-y border-stone-200 bg-white"><div className="mx-auto max-w-7xl px-5 py-20 lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">HOW WE HELP</p><div className="mt-8 grid gap-5 md:grid-cols-3">{highlights.map(([Icon, title, body], index) => <article key={title} className="rounded-2xl border border-stone-200 p-7"><div className="flex justify-between"><span className="font-bold text-[#a57a2c]">0{index + 1}</span><Icon className="text-[#173f3b]"/></div><h3 className="mt-10 text-xl font-bold text-[#173f3b]">{title}</h3><p className="mt-3 leading-7 text-stone-600">{body}</p></article>)}</div></div></section></main></div>;
}
