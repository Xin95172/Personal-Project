"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import { FileText, LayoutPanelTop, MessageCircleQuestion, Save, Trash2 } from "lucide-react";

type Article = { id: string; title: string; excerpt: string; content: string; author_name: string; status: "draft" | "published" };
type Question = { id: string; pseudonym: string; question: string; status: "pending" | "selected" | "answered" | "rejected"; answer: string | null };
type Content = { id: string; page_slug: "home" | "services"; section_key: string; title: string; body: string; cta_label: string; cta_href: string; sort_order: number; status: "draft" | "published" };
type Tab = "articles" | "questions" | "content";

export default function AdminPage() {
  const [tab, setTab] = useState<Tab>("content");
  const [articles, setArticles] = useState<Article[]>([]);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [content, setContent] = useState<Content[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function load() {
    setError("");
    const [a, q, c] = await Promise.all([fetch("/api/admin/articles"), fetch("/api/admin/questions"), fetch("/api/admin/content")]);
    const [ad, qd, cd] = await Promise.all([a.json(), q.json(), c.json()]);
    if (!a.ok || !q.ok || !c.ok) { setError(ad.error ?? qd.error ?? cd.error ?? "Unable to load admin data."); return; }
    setArticles(ad.articles ?? []); setQuestions(qd.questions ?? []); setContent(cd.content ?? []);
  }
  useEffect(() => { void load(); }, []);

  async function request(url: string, method: string, body: unknown) {
    setMessage(""); setError("");
    const response = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) { setError(data.error ?? "Update failed."); return false; }
    setMessage("Saved."); await load(); return true;
  }
  async function createArticle(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); const f = new FormData(event.currentTarget);
    if (await request("/api/admin/articles", "POST", { title: f.get("title"), authorName: f.get("authorName"), excerpt: f.get("excerpt"), content: f.get("content"), status: f.get("status") })) event.currentTarget.reset();
  }
  async function saveContent(event: FormEvent<HTMLFormElement>, id?: string) {
    event.preventDefault(); const f = new FormData(event.currentTarget);
    const body = { id, pageSlug: f.get("pageSlug"), sectionKey: f.get("sectionKey"), title: f.get("title"), body: f.get("body"), ctaLabel: f.get("ctaLabel"), ctaHref: f.get("ctaHref"), sortOrder: f.get("sortOrder"), status: f.get("status") };
    if (await request("/api/admin/content", id ? "PATCH" : "POST", body)) if (!id) event.currentTarget.reset();
  }
  async function deleteItem(url: string, id: string) { if (window.confirm("Delete this item?")) await request(url, "DELETE", { id }); }

  const tabs: { key: Tab; label: string; icon: typeof LayoutPanelTop }[] = [{ key: "content", label: "前台版面", icon: LayoutPanelTop }, { key: "articles", label: "文章／新聞", icon: FileText }, { key: "questions", label: "問答", icon: MessageCircleQuestion }];
  return <main className="min-h-screen bg-[#f5f2eb] px-5 py-10 text-stone-900"><div className="mx-auto max-w-6xl">
    <header className="flex flex-wrap items-end justify-between gap-4 border-b border-stone-300 pb-7"><div><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">SITE ADMIN</p><h1 className="mt-2 text-4xl font-bold text-[#173f3b]">網站內容管理</h1><p className="mt-2 text-stone-600">編輯前台版面、發布文章，以及處理使用者問答。</p></div><Link className="font-bold text-[#173f3b] underline underline-offset-4" href="/">查看前台</Link></header>
    <nav className="mt-6 flex flex-wrap gap-2">{tabs.map(({ key, label, icon: Icon }) => <button key={key} onClick={() => setTab(key)} className={`inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-bold ${tab === key ? "bg-[#173f3b] text-white" : "bg-white text-stone-600"}`}><Icon size={16}/>{label}</button>)}</nav>
    {(message || error) && <p className={`mt-5 rounded-xl px-4 py-3 text-sm ${error ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-800"}`}>{error || message}</p>}
    {tab === "content" && <section className="mt-7 grid gap-7 lg:grid-cols-[.85fr_1.15fr]"><ContentForm onSubmit={saveContent}/><div><h2 className="text-xl font-bold text-[#173f3b]">目前前台內容</h2><p className="mt-1 text-sm text-stone-600">發布後會出現在首頁或服務頁；排序數字較小者優先。</p><div className="mt-4 space-y-4">{content.map(item => <ContentForm key={item.id} item={item} onSubmit={saveContent} onDelete={() => void deleteItem("/api/admin/content", item.id)}/>)}</div></div></section>}
    {tab === "articles" && <section className="mt-7 grid gap-7 lg:grid-cols-2"><form onSubmit={createArticle} className="rounded-2xl bg-white p-6 shadow-sm"><h2 className="text-xl font-bold text-[#173f3b]">新增文章</h2><div className="mt-5 space-y-3"><Field name="title" label="標題" required/><Field name="authorName" label="作者" required/><Field name="excerpt" label="摘要"/><Field name="content" label="內容" required area/><label className="block text-sm font-bold">狀態<select name="status" className="mt-1 w-full rounded-lg border border-stone-300 p-2.5"><option value="draft">草稿</option><option value="published">發布</option></select></label><button className="w-full rounded-lg bg-[#173f3b] py-3 font-bold text-white">儲存文章</button></div></form><div className="space-y-3">{articles.map(a => <article key={a.id} className="rounded-xl bg-white p-5 shadow-sm"><p className="text-xs font-bold text-[#a57a2c]">{a.status} · {a.author_name}</p><h3 className="mt-2 font-bold text-[#173f3b]">{a.title}</h3><p className="mt-2 text-sm text-stone-600">{a.excerpt}</p><div className="mt-4 flex gap-2"><button onClick={() => void request("/api/admin/articles", "PATCH", { id: a.id, status: a.status === "published" ? "draft" : "published" })} className="rounded-md border border-stone-300 px-3 py-1.5 text-xs font-bold">{a.status === "published" ? "改為草稿" : "發布"}</button><button onClick={() => void deleteItem("/api/admin/articles", a.id)} className="rounded-md border border-red-200 p-1.5 text-red-700"><Trash2 size={15}/></button></div></article>)}</div></section>}
    {tab === "questions" && <section className="mt-7 space-y-4">{questions.map(q => <article key={q.id} className="rounded-2xl bg-white p-6 shadow-sm"><p className="text-sm font-bold text-[#a57a2c]">{q.status} · {q.pseudonym}</p><p className="mt-2 font-bold text-[#173f3b]">{q.question}</p><form onSubmit={(e) => { e.preventDefault(); const f = new FormData(e.currentTarget); void request("/api/admin/questions", "PATCH", { id: q.id, status: f.get("status"), answer: f.get("answer") }); }} className="mt-4 grid gap-3"><textarea name="answer" defaultValue={q.answer ?? ""} className="min-h-24 rounded-lg border border-stone-300 p-3" placeholder="回覆內容"/><div className="flex gap-3"><select name="status" defaultValue={q.status} className="rounded-lg border border-stone-300 p-2"><option value="pending">待處理</option><option value="selected">精選</option><option value="answered">已回覆</option><option value="rejected">不採用</option></select><button className="inline-flex items-center gap-2 rounded-lg bg-[#173f3b] px-4 text-sm font-bold text-white"><Save size={15}/>儲存</button></div></form></article>)}{questions.length === 0 && <Empty text="目前沒有提問。"/>}</section>}
  </div></main>;
}

function Field({ name, label, required, area = false, defaultValue = "" }: { name: string; label: string; required?: boolean; area?: boolean; defaultValue?: string }) { const shared = "mt-1 w-full rounded-lg border border-stone-300 p-2.5 font-normal"; return <label className="block text-sm font-bold">{label}{area ? <textarea name={name} required={required} defaultValue={defaultValue} className={`${shared} min-h-32`}/> : <input name={name} required={required} defaultValue={defaultValue} className={shared}/>}</label>; }
function Empty({ text }: { text: string }) { return <p className="rounded-xl border border-dashed border-stone-300 p-8 text-center text-stone-500">{text}</p>; }
function ContentForm({ item, onSubmit, onDelete }: { item?: Content; onSubmit: (event: FormEvent<HTMLFormElement>, id?: string) => Promise<void>; onDelete?: () => void }) { return <form onSubmit={(e) => void onSubmit(e, item?.id)} className="rounded-2xl bg-white p-5 shadow-sm"><div className="grid gap-3 sm:grid-cols-2">{!item && <><label className="text-sm font-bold">頁面<select name="pageSlug" className="mt-1 w-full rounded-lg border border-stone-300 p-2"><option value="home">首頁</option><option value="services">服務頁</option></select></label><Field name="sectionKey" label="區塊代號（英文）" required/></>}<Field name="title" label="標題" required defaultValue={item?.title}/><label className="text-sm font-bold">排序<input name="sortOrder" type="number" defaultValue={item?.sort_order ?? 0} className="mt-1 w-full rounded-lg border border-stone-300 p-2"/></label></div><Field name="body" label="內文" area defaultValue={item?.body}/><div className="mt-3 grid gap-3 sm:grid-cols-2"><Field name="ctaLabel" label="按鈕文字" defaultValue={item?.cta_label}/><Field name="ctaHref" label="按鈕連結" defaultValue={item?.cta_href}/></div><label className="mt-3 block text-sm font-bold">狀態<select name="status" defaultValue={item?.status ?? "draft"} className="mt-1 w-full rounded-lg border border-stone-300 p-2"><option value="draft">草稿</option><option value="published">發布</option></select></label><div className="mt-4 flex gap-2"><button className="inline-flex items-center gap-2 rounded-lg bg-[#173f3b] px-4 py-2.5 text-sm font-bold text-white"><Save size={15}/>{item ? "更新" : "新增區塊"}</button>{onDelete && <button type="button" onClick={onDelete} className="rounded-lg border border-red-200 px-3 text-red-700"><Trash2 size={16}/></button>}</div></form>; }
