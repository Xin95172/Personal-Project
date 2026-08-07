"use client";

import { FormEvent, useState } from "react";
import { ArrowRight, Search } from "lucide-react";

type Trademark = {
  docid?: string;
  tmark_name?: string;
  appl_no?: string;
  appl_date?: string;
  exam_no?: string;
  right_exam_no?: string;
  item_name?: string;
  tmark_class_text?: string;
  full_file_name?: string;
  name_c_text?: string;
  name_e_text?: string;
  goods_class_text?: string;
  goods_class_name_list?: string[] | string;
};

export default function TrademarkSearch() {
  const [keyword, setKeyword] = useState("");
  const [results, setResults] = useState<Trademark[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedKeyword = keyword.trim();

    if (!trimmedKeyword) {
      setError("請輸入商標名稱");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);

    try {
      // 1. 先呼叫你自己的搜尋 API
      const searchResponse = await fetch("/api/trademark/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          keyword: trimmedKeyword,
        }),
      });

      const searchData = await searchResponse.json();

      if (!searchResponse.ok) {
        throw new Error(searchData.error ?? "商標搜尋失敗");
      }

      setTotal(Number(searchData.numFound ?? 0));

      const docs = Array.isArray(searchData.docs)
        ? searchData.docs
        : [];

      if (docs.length === 0) {
        setResults([]);
        return;
      }

      // 先只抓前 20 筆，避免一次抓太多
      const firstPageDocs = docs.slice(0, 20);

      // 2. 再呼叫你自己的 details API
      const detailResponse = await fetch("/api/trademark/details", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          docs: firstPageDocs,
        }),
      });

      const detailData = await detailResponse.json();

      if (!detailResponse.ok) {
        throw new Error(detailData.error ?? "商標詳細資料讀取失敗");
      }

      setResults(Array.isArray(detailData) ? detailData : []);
    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "搜尋時發生未知錯誤"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form
        onSubmit={submitSearch}
        className="rounded-2xl bg-white p-2 shadow-[0_18px_50px_rgba(20,55,51,0.15)]"
      >
        <label htmlFor="trademark" className="sr-only">
          商標名稱
        </label>

        <div className="flex flex-col gap-2 sm:flex-row">
          <div className="flex flex-1 items-center gap-3 px-4 py-3">
            <Search
              className="shrink-0 text-[#a57a2c]"
              size={21}
            />

            <input
              id="trademark"
              value={keyword}
              onChange={(event) =>
                setKeyword(event.target.value)
              }
              placeholder="輸入欲檢索的商標名稱"
              className="w-full bg-transparent text-[15px] text-stone-900 outline-none placeholder:text-stone-400"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#173f3b] px-6 py-3.5 text-sm font-bold text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "搜尋中..." : "搜尋"}

            {!loading && <ArrowRight size={16} />}
          </button>
        </div>

        <p className="px-4 pb-2 pt-1 text-xs text-stone-500">
          商標資料來源為經濟部智慧財產局公開檢索系統。
        </p>
      </form>

      {error && (
        <div className="mt-4 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {!loading && total > 0 && (
        <div className="mt-6">
          <p className="text-sm text-stone-600">
            共找到{" "}
            <span className="font-bold text-[#173f3b]">
              {total.toLocaleString()}
            </span>{" "}
            筆結果
          </p>

          {total > 1000 && (
            <p className="mt-1 text-xs text-amber-700">
              智慧財產局目前僅提供前 1,000 筆搜尋結果，建議增加搜尋條件以縮小範圍。
            </p>
          )}
        </div>
      )}

      <div className="mt-6 space-y-4">
        {results.map((item) => {
          const imageUrl = item.full_file_name
            ? `https://cloud.tipo.gov.tw/S282/S282WV1${item.full_file_name}`
            : null;

          return (
            <article
              key={item.docid ?? item.appl_no}
              className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm"
            >
              <div className="flex flex-col gap-5 sm:flex-row">
                <div className="flex h-36 w-full shrink-0 items-center justify-center overflow-hidden rounded-xl bg-stone-50 sm:w-36">
                  {imageUrl ? (
                    <img
                      src={imageUrl}
                      alt={item.tmark_name ?? "商標圖樣"}
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-stone-400">
                      無圖片
                    </span>
                  )}
                </div>

                <div className="min-w-0 flex-1">
                  <h3 className="text-lg font-bold text-[#173f3b]">
                    {item.tmark_name || "未提供商標名稱"}
                  </h3>

                  <dl className="mt-4 grid gap-2 text-sm text-stone-600">
                    <div>
                      <dt className="inline font-medium text-stone-900">
                        申請案號：
                      </dt>
                      <dd className="inline">
                        {item.appl_no || "—"}
                      </dd>
                    </div>

                    <div>
                      <dt className="inline font-medium text-stone-900">
                        申請日期：
                      </dt>
                      <dd className="inline">
                        {item.appl_date || "—"}
                      </dd>
                    </div>

                    <div>
                      <dt className="inline font-medium text-stone-900">
                        商標種類：
                      </dt>
                      <dd className="inline">
                        {item.tmark_class_text || "—"}
                      </dd>
                    </div>

                    <div>
                      <dt className="inline font-medium text-stone-900">
                        案件狀態：
                      </dt>
                      <dd className="inline">
                        {item.item_name || "—"}
                      </dd>
                    </div>

                    <div>
                      <dt className="inline font-medium text-stone-900">
                        申請人：
                      </dt>
                      <dd className="inline">
                        {item.name_c_text ||
                          item.name_e_text ||
                          "—"}
                      </dd>
                    </div>

                    <div>
                      <dt className="inline font-medium text-stone-900">
                        商品類別：
                      </dt>
                      <dd className="inline">
                        {item.goods_class_text || "—"}
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>
            </article>
          );
        })}
      </div>

      {!loading && total > 0 && results.length === 0 && (
        <p className="mt-6 rounded-xl border border-dashed border-stone-300 p-8 text-center text-stone-500">
          找不到可顯示的商標資料。
        </p>
      )}
    </div>
  );
}