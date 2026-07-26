import { MessageCircleQuestion } from "lucide-react";
import SiteHeader from "@/components/site-header";
import QuestionSubmissionForm from "@/components/question-submission-form";

const answeredQuestions = [
  "商標名稱有相同的字，還能不能申請？",
  "剛開始創業，應該先申請幾個類別？",
  "品牌的英文名稱需要另外申請嗎？",
];

export default function QaPage() {
  return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader/><main>
    <section className="bg-[#173f3b] px-5 py-16 text-[#f7f0df]"><div className="mx-auto max-w-6xl lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#d8b66e]">COMMUNITY Q&A</p><h1 className="mt-4 text-4xl font-bold md:text-5xl">把你的商標問題，交給大家看見。</h1><p className="mt-6 max-w-2xl leading-8 text-stone-300">這裡是互動提問區，與 FAQ 的固定知識整理不同。每收到 20 題投稿，我們將精選 1 題公開回覆，讓更多品牌一起受益。</p></div></section>
    <section className="mx-auto grid max-w-6xl gap-10 px-5 py-16 lg:grid-cols-[1.1fr_.9fr] lg:px-8"><div><div className="flex items-start gap-4 rounded-2xl border border-[#d8b66e]/50 bg-[#f5eddb] p-6"><MessageCircleQuestion className="mt-1 text-[#a57a2c]"/><div><h2 className="font-bold text-[#173f3b]">投稿規則</h2><p className="mt-2 leading-7 text-stone-600">為維持回覆品質，我們以每累積 20 題投稿為一批，挑選其中 1 題回答。投稿不保證個別回覆，也不構成法律意見。</p></div></div><h2 className="mt-10 text-2xl font-bold text-[#173f3b]">已公開的精選提問</h2><div className="mt-5 space-y-3">{answeredQuestions.map((question, index) => <article key={question} className="rounded-xl border border-stone-200 bg-white p-5"><p className="text-xs font-bold tracking-[.12em] text-[#a57a2c]">SELECTED · {String(index + 1).padStart(2, "0")}</p><h3 className="mt-2 font-bold text-[#173f3b]">{question}</h3><p className="mt-3 leading-7 text-stone-600">商標是否可申請，需要綜合指定類別、整體印象與既有商標的使用情況判斷；建議先檢索再評估申請策略。</p></article>)}</div></div>
      <QuestionSubmissionForm />
    </section>
  </main></div>;
}
