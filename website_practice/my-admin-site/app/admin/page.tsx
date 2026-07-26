"use client";

import { FormEvent, useEffect, useState } from "react";
import { Check, FileText, MessageCircleQuestion, Trash2 } from "lucide-react";
import Link from "next/link";

type Article = { id: string; title: string; excerpt: string; content: string; author_name: string; status: "draft" | "published"; created_at: string };
type Question = { id: string; pseudonym: string; question: string; status: "pending" | "selected" | "answered" | "rejected"; answer: string | null; created_at: string };

export default function AdminPage() {
  const [tab, setTab] = useState<"articles" | "questions">("articles");
  const [articles, setArticles] = useState<Article[]>([]);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function load() {
    setError("");
    const [articleResponse, questionResponse] = await Promise.all([fetch("/api/admin/articles"), fetch("/api/admin/questions")]);
    const articleData = await articleResponse.json();
    const questionData = await questionResponse.json();
    if (!articleResponse.ok) { setError(articleData.error ?? "無法讀取管理資料"); return; }
    setArticles(articleData.articles ?? []);
    if (questionResponse.ok) setQuestions(questionData.questions ?? []);
  }

  useEffect(() => { void load(); }, []);

  async function createArticle(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true); setMessage(""); setError("");
    const form = new FormData(event.currentTarget);
    const response = await fetch("/api/admin/articles", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: form.get("title"), authorName: form.get("authorName"), excerpt: form.get("excerpt"), content: form.get("content"), status: form.get("status") }) });
    const data = await response.json();
    setSubmitting(false);
    if (!response.ok) { setError(data.error ?? "文章儲存失敗"); return; }
    event.currentTarget.reset(); setMessage("文章已儲存"); await load();
  }

  async function updateArticle(id: string, status: "draft" | "published") {
    const response = await fetch("/api/admin/articles", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id, status }) });
    const data = await response.json();
    if (!response.ok) setError(data.error ?? "文章更新失敗"); else { setMessage(status === "published" ? "文章已發布" : "文章已改為草稿"); await load(); }
  }

  async function deleteArticle(id: string) {
    if (!window.confirm("確定刪除這篇文章嗎？")) return;
    const response = await fetch("/api/admin/articles", { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }) });
    const data = await response.json();
    if (!response.ok) setError(data.error ?? "文章刪除失敗"); else { setMessage("文章已刪除"); await load(); }
  }

  async function answerQuestion(event: FormEvent<HTMLFormElement>, id: string, status: Question["status"]) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const response = await fetch("/api/admin/questions", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id, status, answer: form.get("answer") }) });
    const data = await response.json();
    if (!response.ok) setError(data.error ?? "問答更新失敗"); else { setMessage("問答狀態已更新"); await load(); }
  }

  return <main className="min-h-screen bg-[#f5f2eb] px-5 py-10 text-stone-900"><div className="mx-auto max-w-6xl"><header className="flex flex-col justify-between gap-5 border-b border-stone-300 pb-8 md:flex-row md:items-end"><div><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">XX ADMIN</p><h1 className="mt-3 text-4xl font-bold text-[#173f3b]">網站內容管理</h1><p className="mt-3 text-stone-600">管理專欄文章與互動問答投稿。</p></div><Link href="/" className="text-sm font-bold text-[#173f3b] underline underline-offset-4">回到網站前台</Link></header>
    <div className="mt-7 flex gap-2"><button onClick={() => setTab("articles")} className={`rounded-full px-5 py-2.5 text-sm font-bold ${tab === "articles" ? "bg-[#173f3b] text-white" : "bg-white text-stone-600"}`}>專欄文章</button><button onClick={() => setTab("questions")} className={`rounded-full px-5 py-2.5 text-sm font-bold ${tab === "questions" ? "bg-[#173f3b] text-white" : "bg-white text-stone-600"}`}>互動問答 <span className="ml-1 text-xs opacity-75">{questions.filter((item) => item.status === "pending").length}</span></button></div>
    {(message || error) && <p className={`mt-5 rounded-xl px-4 py-3 text-sm ${error ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-800"}`}>{error || message}</p>}
    {tab === "articles" ? <section className="mt-7 grid gap-7 lg:grid-cols-[.92fr_1.08fr]"><form onSubmit={createArticle} className="h-fit rounded-2xl bg-white p-6 shadow-sm"><div className="flex items-center gap-2 text-[#173f3b]"><FileText size={20}/><h2 className="text-xl font-bold">新增專欄文章</h2></div><div className="mt-6 space-y-4"><label className="block text-sm font-bold">文章標題<input name="title" required className="mt-2 w-full rounded-lg border border-stone-300 px-3 py-2.5 font-normal"/></label><label className="block text-sm font-bold">作者名稱<input name="authorName" required className="mt-2 w-full rounded-lg border border-stone-300 px-3 py-2.5 font-normal"/></label><label className="block text-sm font-bold">文章摘要<input name="excerpt" className="mt-2 w-full rounded-lg border border-stone-300 px-3 py-2.5 font-normal"/></label><label className="block text-sm font-bold">文章內容<textarea name="content" required className="mt-2 min-h-44 w-full rounded-lg border border-stone-300 px-3 py-2.5 font-normal"/></label><label className="block text-sm font-bold">儲存狀態<select name="status" className="mt-2 w-full rounded-lg border border-stone-300 px-3 py-2.5 font-normal"><option value="draft">草稿</option><option value="published">直接發布</option></select></label><button disabled={submitting} className="w-full rounded-lg bg-[#173f3b] py-3 text-sm font-bold text-white disabled:opacity-50">{submitting ? "儲存中…" : "儲存文章"}</button></div></form><div><h2 className="text-xl font-bold text-[#173f3b]">現有文章</h2><div className="mt-4 space-y-3">{articles.map((article) => <article key={article.id} className="rounded-xl bg-white p-5 shadow-sm"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs font-bold text-[#a57a2c]">{article.status === "published" ? "已發布" : "草稿"} · {article.author_name}</p><h3 className="mt-2 font-bold text-[#173f3b]">{article.title}</h3></div><div className="flex gap-2"><button onClick={() => void updateArticle(article.id, article.status === "published" ? "draft" : "published")} className="rounded-md border border-stone-300 px-3 py-1.5 text-xs font-bold">{article.status === "published" ? "改為草稿" : "發布"}</button><button onClick={() => void deleteArticle(article.id)} className="rounded-md border border-red-200 p-1.5 text-red-700"><Trash2 size={15}/></button></div></div>{article.excerpt && <p className="mt-3 text-sm leading-6 text-stone-600">{article.excerpt}</p>}</article>)}{articles.length === 0 && <p className="rounded-xl border border-dashed border-stone-300 p-8 text-center text-stone-500">尚未建立文章。</p>}</div></div></section> : <section className="mt-7"><div className="flex items-center gap-2 text-[#173f3b]"><MessageCircleQuestion size={21}/><h2 className="text-xl font-bold">互動問答投稿</h2></div><p className="mt-2 text-sm text-stone-600">每累積 20 題投稿，請精選 1 題回覆並設為「已回答」。</p><div className="mt-5 space-y-4">{questions.map((question) => <article key={question.id} className="rounded-2xl bg-white p-6 shadow-sm"><div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm font-bold text-[#a57a2c]">{question.status} · {question.pseudonym}</p><select defaultValue={question.status} onChange={(event) => void answerQuestion({ preventDefault() {}, currentTarget: event.currentTarget.form! } as unknown as FormEvent<HTMLFormElement>, question.id, event.target.value as Question["status"])} className="rounded-md border border-stone-300 px-2 py-1 text-sm"><option value="pending">待處理</option><option value="selected">已精選</option><option value="answered">已回答</option><option value="rejected">不採用</option></select></div><p className="mt-3 font-medium text-[#173f3b]">{question.question}</p><form onSubmit={(event) => void answerQuestion(event, question.id, "answered")} className="mt-4"><textarea name="answer" defaultValue={question.answer ?? ""} className="min-h-24 w-full rounded-lg border border-stone-300 px-3 py-2 text-sm" placeholder="輸入公開回覆內容"/><button className="mt-2 inline-flex items-center gap-2 rounded-lg bg-[#173f3b] px-4 py-2 text-sm font-bold text-white"><Check size={15}/>儲存並發布回覆</button></form></article>)}{questions.length === 0 && <p className="rounded-xl border border-dashed border-stone-300 p-8 text-center text-stone-500">尚無投稿。</p>}</div></section>}
  </div></main>;
}
