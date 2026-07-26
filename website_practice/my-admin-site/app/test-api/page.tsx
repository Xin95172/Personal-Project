"use client";

import { useState } from "react";

type ApiResult = {
  success: boolean;
  message: string;
  serverTime: string;
};

export default function TestApiPage() {
  const [result, setResult] = useState<ApiResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function callApi() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/hello");

      if (!response.ok) {
        throw new Error(`API 錯誤：${response.status}`);
      }

      const data = (await response.json()) as ApiResult;
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "發生未知錯誤",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-xl p-8">
      <h1 className="text-3xl font-bold">
        前後端連線測試
      </h1>

      <p className="mt-3 text-gray-600">
        按下按鈕後，前台會呼叫後端 API。
      </p>

      <button
        type="button"
        onClick={callApi}
        disabled={loading}
        className="mt-6 rounded bg-black px-5 py-3 text-white disabled:opacity-50"
      >
        {loading ? "呼叫中……" : "呼叫後端 API"}
      </button>

      {error && (
        <p className="mt-6 rounded border border-red-300 p-4">
          {error}
        </p>
      )}

      {result && (
        <section className="mt-6 rounded border p-4">
          <p>
            成功：{result.success ? "是" : "否"}
          </p>

          <p>訊息：{result.message}</p>

          <p>伺服器時間：{result.serverTime}</p>
        </section>
      )}
    </main>
  );
}