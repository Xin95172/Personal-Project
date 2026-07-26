"use client";

import { FormEvent, useState } from "react";
import { ArrowUpRight, Search } from "lucide-react";

const tipoSearchUrl = "https://cloud.tipo.gov.tw/S282/S282WV1/";

export default function TrademarkSearch() {
  const [keyword, setKeyword] = useState("");

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    window.open(tipoSearchUrl, "_blank", "noopener,noreferrer");
  }

  return (
    <form onSubmit={submitSearch} className="rounded-2xl bg-white p-2 shadow-[0_18px_50px_rgba(20,55,51,0.15)]">
      <label htmlFor="trademark" className="sr-only">商標名稱</label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <div className="flex flex-1 items-center gap-3 px-4 py-3">
          <Search className="shrink-0 text-[#a57a2c]" size={21} />
          <input
            id="trademark"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            placeholder="輸入欲檢索的商標名稱"
            className="w-full bg-transparent text-[15px] text-stone-900 outline-none placeholder:text-stone-400"
          />
        </div>
        <button type="submit" className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#173f3b] px-6 py-3.5 text-sm font-bold text-white transition hover:bg-[#0d302d]">
          前往檢索 <ArrowUpRight size={16} />
        </button>
      </div>
      <p className="px-4 pb-2 pt-1 text-xs text-stone-500">將開啟經濟部智慧財產局商標檢索系統，建議先以文字近似搜尋確認。</p>
    </form>
  );
}
