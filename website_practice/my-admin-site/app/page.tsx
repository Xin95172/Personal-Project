import Link from "next/link";
import { ArrowRight, FileText, Search, ShieldCheck, Sparkles } from "lucide-react";
import SiteHeader from "@/components/site-header";
import TrademarkSearch from "@/components/trademark-search";

const services = [
  { no: "01", title: "商標檢索與評估", text: "從文字、圖樣到指定類別，先釐清近似風險，再規劃下一步。", icon: Search },
  { no: "02", title: "商標申請服務", text: "協助分類、準備文件與送件流程，讓申請更有依據、時程更清楚。", icon: FileText },
  { no: "03", title: "品牌權利守護", text: "從註冊後管理到爭議初步判讀，陪伴品牌累積能被保護的價值。", icon: ShieldCheck },
];

const questions = [
  "商標一定要先申請才能使用嗎？", "商標申請到核准需要多久？", "名稱相同就一定不能申請嗎？", "一個商標可以保護哪些商品或服務？"
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#fbfaf7] text-stone-900">
      <SiteHeader />

      <main>
        <section className="relative overflow-hidden bg-[#e9e3d5]">
          <div className="absolute -right-20 top-0 h-full w-[43%] bg-[#d9c39a]/35" />
          <div className="absolute right-[14%] top-16 hidden h-72 w-72 rounded-full border border-[#173f3b]/15 lg:block" />
          <div className="relative mx-auto grid max-w-7xl gap-14 px-5 py-20 lg:grid-cols-[1.15fr_.85fr] lg:px-8 lg:py-28">
            <div className="max-w-3xl">
              <p className="mb-6 flex items-center gap-2 text-sm font-bold tracking-[0.16em] text-[#9c6c1c]"><span className="h-px w-8 bg-[#9c6c1c]" />TRADEMARK COUNSEL</p>
              <h1 className="display-serif text-5xl leading-[1.18] tracking-tight text-[#173f3b] md:text-6xl lg:text-7xl">
                讓每一個名字，<br />都有被守護的權利。
              </h1>
              <p className="mt-7 max-w-xl text-base leading-8 text-stone-600 md:text-lg">從商標檢索、申請到品牌權利規劃，XX 以清楚的說明與細緻的流程，陪您讓品牌穩健成長。</p>
              <div className="mt-10 flex flex-wrap gap-3">
                <Link href="/services" className="inline-flex items-center gap-2 rounded-full bg-[#173f3b] px-6 py-3.5 text-sm font-bold text-white transition hover:bg-[#0d302d]">先了解服務 <ArrowRight size={17} /></Link>
                <Link href="/contact" className="rounded-full border border-[#173f3b]/30 bg-white/60 px-6 py-3.5 text-sm font-bold text-[#173f3b] transition hover:border-[#173f3b]">預約初步諮詢</Link>
              </div>
            </div>
            <div className="relative flex items-center justify-center lg:justify-end">
              <div className="relative grid aspect-square w-full max-w-[360px] place-items-center rounded-full border-[18px] border-[#f7f0df] bg-[#173f3b] shadow-2xl shadow-[#173f3b]/15">
                <div className="absolute inset-5 rounded-full border border-[#d8b66e]/60" />
                <div className="relative text-center text-[#f7f0df]"><p className="display-serif text-7xl">衡</p><p className="mt-3 text-xs font-semibold tracking-[0.3em]">BRAND · RIGHTS · TRUST</p></div>
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-5 py-20 lg:px-8">
          <div className="grid gap-10 lg:grid-cols-[.82fr_1.18fr] lg:items-end">
            <div><p className="text-sm font-bold tracking-[0.15em] text-[#a57a2c]">TRADEMARK SEARCH</p><h2 className="mt-4 text-3xl font-bold tracking-tight text-[#173f3b] md:text-4xl">先查詢，再安心前進。</h2></div>
            <p className="max-w-2xl leading-7 text-stone-600">商標申請前，檢索是降低近似與權利衝突風險的重要起點。輸入名稱後，即可前往智慧財產局的公開資料庫進一步查詢。</p>
          </div>
          <div className="mt-9 rounded-[2rem] bg-[#d8b66e] p-5 md:p-8"><TrademarkSearch /></div>
        </section>

        <section className="border-y border-stone-200 bg-white">
          <div className="mx-auto max-w-7xl px-5 py-20 lg:px-8">
            <div className="flex flex-wrap items-end justify-between gap-5"><div><p className="text-sm font-bold tracking-[0.15em] text-[#a57a2c]">HOW WE HELP</p><h2 className="mt-3 text-3xl font-bold text-[#173f3b]">把商標這件事，說得簡單一點。</h2></div><Link href="/services" className="inline-flex items-center gap-2 text-sm font-bold text-[#173f3b]">查看完整服務 <ArrowRight size={16} /></Link></div>
            <div className="mt-10 grid gap-5 md:grid-cols-3">{services.map(({ no, title, text, icon: Icon }) => <article key={no} className="group rounded-2xl border border-stone-200 p-7 transition hover:-translate-y-1 hover:border-[#d8b66e] hover:shadow-lg hover:shadow-stone-200/50"><div className="flex items-start justify-between"><span className="text-sm font-bold text-[#a57a2c]">{no}</span><Icon size={23} className="text-[#173f3b]" /></div><h3 className="mt-12 text-xl font-bold text-[#173f3b]">{title}</h3><p className="mt-3 leading-7 text-stone-600">{text}</p></article>)}</div>
          </div>
        </section>

        <section className="mx-auto grid max-w-7xl gap-10 px-5 py-20 lg:grid-cols-2 lg:px-8">
          <div className="rounded-[2rem] bg-[#173f3b] p-8 text-[#f7f0df] md:p-10"><Sparkles className="text-[#d8b66e]" /><p className="mt-10 text-sm font-bold tracking-[0.15em] text-[#d8b66e]">START HERE</p><h2 className="mt-4 text-3xl font-bold leading-tight">不確定從哪一步開始？<br />先認識商標分類。</h2><p className="mt-5 max-w-md leading-7 text-stone-300">商品或服務類別會影響保護範圍。透過淺顯指引，先找到您的品牌需要的位置。</p><Link href="/services" className="mt-9 inline-flex items-center gap-2 text-sm font-bold text-[#f7f0df] underline decoration-[#d8b66e] underline-offset-8">閱讀分類說明 <ArrowRight size={16} /></Link></div>
          <div className="rounded-[2rem] border border-stone-200 bg-[#f4f1e9] p-8 md:p-10"><p className="text-sm font-bold tracking-[0.15em] text-[#a57a2c]">COMMUNITY Q&A</p><h2 className="mt-4 text-3xl font-bold text-[#173f3b]">最新互動問答</h2><div className="mt-6 divide-y divide-stone-300">{questions.map((question, index) => <Link href="/qa" key={question} className="flex items-center justify-between gap-5 py-4 text-[15px] font-medium text-stone-700 transition hover:text-[#173f3b]"><span><span className="mr-3 text-xs font-bold text-[#a57a2c]">0{index + 1}</span>{question}</span><ArrowRight size={16} className="shrink-0" /></Link>)}</div><Link href="/qa" className="mt-7 inline-flex items-center gap-2 text-sm font-bold text-[#173f3b]">每 20 題精選回覆 1 題 <ArrowRight size={16} /></Link></div>
        </section>

        <section className="bg-[#eee8da]"><div className="mx-auto flex max-w-7xl flex-col justify-between gap-7 px-5 py-14 md:flex-row md:items-center lg:px-8"><div><p className="text-sm font-bold tracking-[0.15em] text-[#a57a2c]">LET&apos;S TALK</p><h2 className="mt-3 text-3xl font-bold text-[#173f3b]">準備好讓品牌被正式看見了嗎？</h2></div><Link href="/contact" className="inline-flex w-fit items-center gap-2 rounded-full bg-[#173f3b] px-6 py-3.5 text-sm font-bold text-white">開始申請服務 <ArrowRight size={17} /></Link></div></section>
      </main>
      <footer className="bg-[#102c29] px-5 py-10 text-stone-300"><div className="mx-auto flex max-w-7xl flex-col justify-between gap-4 text-sm md:flex-row lg:px-3"><p className="font-semibold text-[#f7f0df]">XX · TRADEMARK SERVICES</p><p>以專業守護每個值得被記住的名字。</p></div></footer>
    </div>
  );
}
