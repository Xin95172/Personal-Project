"use client";

import { FormEvent, useState } from "react";
import { ArrowRight } from "lucide-react";

export default function QuestionSubmissionForm() {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setLoading(true); setMessage("");
    const form = new FormData(event.currentTarget);
    const response = await fetch("/api/questions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pseudonym: form.get("pseudonym"), question: form.get("question") }) });
    const data = await response.json(); setLoading(false);
    if (!response.ok) { setMessage(data.error ?? "投稿失敗，請稍後再試。"); return; }
    event.currentTarget.reset(); setMessage("投稿已送出，謝謝你的提問！");
  }

  return <form onSubmit={submit} className="h-fit rounded-3xl border border-stone-200 bg-white p-7 shadow-sm"><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">SUBMIT A QUESTION</p><h2 className="mt-3 text-2xl font-bold text-[#173f3b]">投稿你的問題</h2><p className="mt-3 text-sm leading-6 text-stone-600">投稿將直接送入後台，供編輯團隊整理與精選。</p><label className="mt-7 block text-sm font-bold text-stone-700">暱稱（選填）<input name="pseudonym" className="mt-2 w-full rounded-lg border border-stone-300 px-4 py-3 font-normal outline-none focus:border-[#173f3b]" placeholder="例如：正在創業的人"/></label><label className="mt-5 block text-sm font-bold text-stone-700">你的問題<textarea name="question" required className="mt-2 min-h-36 w-full rounded-lg border border-stone-300 px-4 py-3 font-normal outline-none focus:border-[#173f3b]" placeholder="請描述你的商標問題，不要填入個人敏感資料。"/></label><button disabled={loading} className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-[#173f3b] py-3.5 text-sm font-bold text-white disabled:opacity-60">{loading ? "送出中…" : <>送出投稿 <ArrowRight size={16}/></>}</button>{message && <p className="mt-4 text-sm text-[#173f3b]">{message}</p>}</form>;
}
