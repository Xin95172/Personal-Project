import Link from "next/link";

export default function PaymentCheckoutPage() {
  return (
    <main className="min-h-screen bg-stone-50 px-5 py-16">
      <div className="mx-auto max-w-lg rounded-2xl bg-white p-8 shadow-sm">
        <h1 className="text-3xl font-bold text-stone-900">付款功能尚未啟用</h1>
        <p className="mt-4 text-stone-600">
          此頁面保留給未來的付款流程，目前不會建立或處理任何付款。
        </p>
        <Link href="/" className="mt-6 inline-block font-semibold text-[#173f3b] underline underline-offset-4">
          返回首頁
        </Link>
      </div>
    </main>
  );
}
