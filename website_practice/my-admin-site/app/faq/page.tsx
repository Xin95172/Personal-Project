import SiteHeader from "@/components/site-header";

const faqs = [
  "商標一定要先申請才能使用嗎？", "商標申請到核准需要多久？", "名稱相同就一定不能申請嗎？", "一個商標可以保護哪些商品或服務？", "文字商標與圖樣商標有什麼不同？", "商標分類要如何選擇？", "可以一次申請多個類別嗎？", "個人也可以申請商標嗎？", "公司名稱等於商標嗎？", "商標註冊後還需要注意什麼？", "商標可以轉讓給他人嗎？", "商標可以授權他人使用嗎？", "申請中可以使用 ® 嗎？", "什麼是商標近似？", "商標被核駁後還有機會嗎？", "海外品牌應如何保護？", "商標註冊有效多久？", "未使用商標會有什麼影響？", "網路帳號名稱能受商標保護嗎？", "何時該尋求專業協助？"
];

export default function FaqPage() {
  return <div className="min-h-screen bg-[#fbfaf7]"><SiteHeader/><main className="mx-auto max-w-4xl px-5 py-16 lg:px-8"><p className="text-sm font-bold tracking-[.15em] text-[#a57a2c]">TRADEMARK Q&A</p><h1 className="mt-4 text-4xl font-bold text-[#173f3b]">商標常見問答</h1><p className="mt-5 max-w-2xl leading-8 text-stone-600">用 20 個最常見的問題，陪您快速建立商標的基本概念。個別案件仍應依實際情況評估。</p><div className="mt-10 divide-y divide-stone-200 rounded-2xl border border-stone-200 bg-white">{faqs.map((faq, index) => <details key={faq} className="group px-6 py-5"><summary className="flex cursor-pointer list-none items-center gap-5 font-bold text-[#173f3b]"><span className="text-sm text-[#a57a2c]">{String(index + 1).padStart(2, "0")}</span>{faq}<span className="ml-auto text-xl font-normal transition group-open:rotate-45">+</span></summary><p className="ml-9 mt-4 max-w-2xl leading-7 text-stone-600">此題的答案會依商標內容、使用方式與指定類別而有所不同。建議先完成基本檢索，再由專業人員依您的品牌情境說明可行方向。</p></details>)}</div></main></div>;
}
