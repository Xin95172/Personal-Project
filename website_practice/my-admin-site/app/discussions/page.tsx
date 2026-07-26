import Link from "next/link";
import { CalendarDays, MessageSquareText } from "lucide-react";
import SiteHeader from "@/components/site-header";

const topics = [
  { title: "商標申請前，檢索結果應如何解讀？", status: "討論中", period: "2026-07-20 — 2026-08-20", text: "討論文字近似、類別相同與整體印象之間的關係，以及申請前應如何看待公開檢索資料。" },
  { title: "商標使用證據的保存與提出原則", status: "即將開始", period: "2026-08-01 — 2026-08-31", text: "整理商標實際使用時可保留哪些紀錄，以及不同商業情境下的常見問題。" },
  { title: "品牌名稱與商標權：創業初期的選擇", status: "已結束", period: "2026-06-01 — 2026-06-30", text: "已完成討論摘要，包含名稱選擇、類別規劃與常見的替代方案。" },
];

export default function DiscussionsPage() { return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader/><main className="mx-auto max-w-6xl px-5 py-16 lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">TRADEMARK DISCUSSION</p><h1 className="mt-4 text-4xl font-bold text-[#173f3b]">商標權公共討論</h1><p className="mt-5 max-w-3xl leading-8 text-stone-600">先理解商標制度與資料，再以支持、反對或中立立場，提出理由、問題、證據與替代方案。這裡不以聲量或按讚排行作為主要機制。</p><div className="mt-8 rounded-xl border border-[#d8b66e]/60 bg-[#f7f1e5] p-5 text-sm leading-7 text-stone-700">平台討論與整理僅代表參與網站者的意見，不代表全體國民，也不是具有統計代表性的民意調查。</div><div className="mt-10 space-y-4">{topics.map((topic, index) => <article key={topic.title} className="rounded-2xl border border-stone-200 bg-white p-6"><div className="flex flex-wrap items-center gap-3"><span className="rounded-full bg-[#edf2f1] px-3 py-1 text-xs font-bold text-[#173f3b]">{topic.status}</span><span className="flex items-center gap-1 text-sm text-stone-500"><CalendarDays size={14}/>{topic.period}</span></div><h2 className="mt-4 text-xl font-bold text-[#173f3b]">{topic.title}</h2><p className="mt-3 max-w-3xl leading-7 text-stone-600">{topic.text}</p><Link href={index === 0 ? "/discussions/trademark-search" : "/discussions"} className="mt-5 inline-flex items-center gap-2 text-sm font-bold text-[#173f3b]"><MessageSquareText size={16}/>查看議題與討論</Link></article>)}</div></main></div>; }
