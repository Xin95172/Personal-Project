import Link from "next/link";
import { ArrowRight, CheckCircle2 } from "lucide-react";
import SiteHeader from "@/components/site-header";

const serviceGroups = [
  ["商標檢索與風險評估", ["文字、圖樣與近似商標初步檢索", "指定商品／服務類別建議", "申請可行性與權利衝突說明"]],
  ["商標申請與案件管理", ["申請文件準備與電子送件", "審查意見與補正程序協助", "案件進度追蹤與註冊後提醒"]],
  ["品牌權利規劃", ["商標分類與保護範圍盤點", "國內外申請方向初步建議", "商標爭議與使用情境初步諮詢"]],
];

export default function ServicesPage() {
  return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader /><main>
    <section className="bg-[#173f3b] px-5 py-20 text-[#f7f0df]"><div className="mx-auto max-w-7xl lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#d8b66e]">OUR SERVICES</p><h1 className="mt-4 text-4xl font-bold md:text-5xl">商標服務，從釐清到守護。</h1><p className="mt-6 max-w-2xl leading-8 text-stone-300">每個品牌的階段不同。我們以清楚流程協助您確認方向，讓商標不只是申請文件，而是品牌資產的一部分。</p></div></section>
    <section className="mx-auto max-w-7xl px-5 py-16 lg:px-8"><div className="grid gap-6 lg:grid-cols-3">{serviceGroups.map(([title, points], index) => <article key={title as string} className="rounded-2xl border border-stone-200 bg-white p-8"><span className="text-sm font-bold text-[#a57a2c]">0{index + 1}</span><h2 className="mt-10 text-2xl font-bold text-[#173f3b]">{title}</h2><ul className="mt-6 space-y-4">{(points as string[]).map(point => <li key={point} className="flex gap-3 leading-7 text-stone-600"><CheckCircle2 size={18} className="mt-1 shrink-0 text-[#a57a2c]" />{point}</li>)}</ul></article>)}</div>
    <div className="mt-14 rounded-3xl bg-[#eee8da] p-8 md:flex md:items-center md:justify-between"><div><h2 className="text-2xl font-bold text-[#173f3b]">不知道該選哪一項服務？</h2><p className="mt-2 text-stone-600">先告訴我們您的品牌現況，我們會協助您判斷合適的起點。</p></div><Link href="/contact" className="mt-5 inline-flex items-center gap-2 rounded-full bg-[#173f3b] px-6 py-3 font-bold text-white md:mt-0">申請初步諮詢 <ArrowRight size={16}/></Link></div></section>
  </main></div>;
}
