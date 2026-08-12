"use client";

import { useState } from "react";

export default function PaymentTestPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function startPayment() {
    try {
      setLoading(true);
      setError("");

      const response = await fetch("/api/payment/ecpay/create", {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error ?? "建立付款失敗");
      }

      const form = document.createElement("form");

      form.method = "POST";
      form.action = data.action;

      Object.entries(data.params).forEach(([key, value]) => {
        const input = document.createElement("input");

        input.type = "hidden";
        input.name = key;
        input.value = String(value);

        form.appendChild(input);
      });

      document.body.appendChild(form);
      form.submit();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "付款發生錯誤"
      );

      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-stone-50 px-5 py-16">
      <div className="mx-auto max-w-lg rounded-2xl bg-white p-8 shadow-sm">
        <p className="text-sm font-bold text-amber-700">
          ECPay Stage 測試
        </p>

        <h1 className="mt-3 text-3xl font-bold text-stone-900">
          NT$1 金流測試商品
        </h1>

        <p className="mt-3 text-stone-600">
          此頁僅用於綠界測試環境，不會使用正式金流。
        </p>

        <div className="mt-8 flex items-end justify-between border-t pt-6">
          <span className="text-stone-500">測試價格</span>

          <strong className="text-2xl text-stone-900">
            NT$5
          </strong>
        </div>

        {error && (
          <p className="mt-5 rounded-lg bg-red-50 p-3 text-sm text-red-700">
            {error}
          </p>
        )}

        <button
          onClick={startPayment}
          disabled={loading}
          className="mt-6 w-full rounded-xl bg-[#173f3b] py-3.5 font-bold text-white disabled:opacity-50"
        >
          {loading ? "準備付款中..." : "前往綠界測試付款"}
        </button>
      </div>
    </main>
  );
}