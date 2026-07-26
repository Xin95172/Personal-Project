import Link from "next/link";
import { ArrowRight } from "lucide-react";
import SiteHeader from "@/components/site-header";

const articles = [
  ["商標申請前，為什麼一定要先做檢索？", "申請前的檢索能協助辨識文字或圖樣近似的既有案件，為品牌決策保留更多空間。"],
  ["品牌剛起步，商標分類怎麼開始看？", "從您現在提供的商品或服務出發，再預留合理的發展方向，逐步建立保護範圍。"],
  ["公司名稱、網域與商標：三者的差異", "三種名稱各自具有不同的功能與保護機制，及早規劃才能減少後續調整成本。"],
];
export default function NewsPage() { return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader/><main className="mx-auto max-w-6xl px-5 py-16 lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">TRADEMARK NOTES</p><h1 className="mt-4 text-4xl font-bold text-[#173f3b]">商標新聞與觀點</h1><div className="mt-10 grid gap-6 md:grid-cols-3">{articles.map(([title, intro], i) => <article className="rounded-2xl border border-stone-200 bg-white p-7" key={title}><p className="text-sm font-bold text-[#a57a2c]">INSIGHT · 0{i+1}</p><h2 className="mt-12 text-xl font-bold leading-8 text-[#173f3b]">{title}</h2><p className="mt-4 leading-7 text-stone-600">{intro}</p><Link href="/contact" className="mt-8 inline-flex items-center gap-2 text-sm font-bold text-[#173f3b]">與我們討論 <ArrowRight size={16}/></Link></article>)}</div></main></div>; }
