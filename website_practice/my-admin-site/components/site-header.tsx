import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

export default function SiteHeader() {
  const bookmarks = [
    ["/", "首頁"],
    ["/services", "服務項目"],
    ["/faq", "FAQ"],
    ["/qa", "互動問答"],
    ["/columns", "專欄"],
    ["/discussions", "商標討論"],
    ["/tools", "實用工具"],
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-stone-200/80 bg-[#fbfaf7]/95 shadow-sm shadow-stone-950/5 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
        <Link
          href="/"
          className="flex items-center gap-3 text-stone-950"
        >
          <span className="grid h-10 w-10 place-items-center rounded-full bg-[#173f3b] font-serif text-lg font-semibold text-[#f7f0df]">
            XX
          </span>
          <span className="leading-tight">
            <span className="block text-base font-bold tracking-[0.13em]">XX</span>
            <span className="block text-[10px] tracking-[0.18em] text-stone-500">TRADEMARK SERVICES</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-5 text-sm font-medium lg:flex">
          {bookmarks.map(([href, label]) => (
            <Link key={href} href={href} className="whitespace-nowrap text-stone-600 transition hover:text-[#173f3b]">
              {label}
            </Link>
          ))}
        </nav>

        <Link href="/contact" className="inline-flex items-center gap-1.5 rounded-full bg-[#173f3b] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#0d302d]">
          申請諮詢 <ArrowUpRight size={15} />
        </Link>
      </div>
      <nav aria-label="頁面書籤" className="flex gap-2 overflow-x-auto border-t border-stone-200/70 px-5 py-2.5 lg:hidden">
        {bookmarks.map(([href, label]) => (
          <Link key={href} href={href} className="shrink-0 rounded-full bg-stone-100 px-3 py-1.5 text-xs font-semibold text-stone-700 transition hover:bg-[#173f3b] hover:text-white">
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
