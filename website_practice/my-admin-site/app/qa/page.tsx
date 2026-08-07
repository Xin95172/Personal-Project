import { MessageCircleQuestion } from "lucide-react";
import SiteHeader from "@/components/site-header";
import QuestionSubmissionForm from "@/components/question-submission-form";
import { createClient } from "@/lib/supabase/server";
import { connection } from "next/server";

export default async function QaPage() {
  await connection();
  const supabase = await createClient();
  const { data: questions } = await supabase.from("question_submissions").select("id, question, answer, answered_at").eq("status", "answered").not("answer", "is", null).order("answered_at", { ascending: false }).limit(20);
  return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader/><main><section className="bg-[#173f3b] px-5 py-16 text-[#f7f0df]"><div className="mx-auto max-w-6xl lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#d8b66e]">COMMUNITY Q&A</p><h1 className="mt-4 text-4xl font-bold md:text-5xl">商標問題，公開回答</h1><p className="mt-6 max-w-2xl leading-8 text-stone-300">精選問答由管理員審核與發布；歡迎留下你的問題。</p></div></section><section className="mx-auto grid max-w-6xl gap-10 px-5 py-16 lg:grid-cols-[1.1fr_.9fr] lg:px-8"><div><div className="flex items-start gap-4 rounded-2xl border border-[#d8b66e]/50 bg-[#f5eddb] p-6"><MessageCircleQuestion className="mt-1 text-[#a57a2c]"/><div><h2 className="font-bold text-[#173f3b]">精選問答</h2><p className="mt-2 leading-7 text-stone-600">下列內容由管理後台的「問答」分頁發布與更新。</p></div></div><div className="mt-8 space-y-3">{questions?.map((item, index) => <article key={item.id} className="rounded-xl border border-stone-200 bg-white p-5"><p className="text-xs font-bold tracking-[.12em] text-[#a57a2c]">ANSWERED · {String(index + 1).padStart(2, "0")}</p><h3 className="mt-2 font-bold text-[#173f3b]">{item.question}</h3><p className="mt-3 whitespace-pre-line leading-7 text-stone-600">{item.answer}</p></article>)}{!questions?.length && <p className="rounded-xl border border-dashed border-stone-300 p-8 text-center text-stone-500">目前尚無已發布問答。</p>}</div></div><QuestionSubmissionForm/></section></main></div>;
}
