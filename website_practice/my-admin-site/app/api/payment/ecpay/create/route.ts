import { NextResponse } from "next/server";
import { generateCheckMacValue } from "@/lib/ecpay";

function getTradeDate() {
  const now = new Date();

  const parts = new Intl.DateTimeFormat("zh-TW", {
    timeZone: "Asia/Taipei",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(now);

  const get = (type: string) =>
    parts.find((item) => item.type === type)?.value ?? "";

  return `${get("year")}/${get("month")}/${get("day")} ${get("hour")}:${get("minute")}:${get("second")}`;
}

function createMerchantTradeNo() {
  // 綠界編號需唯一，這裡控制在 20 字元內
  return `T${Date.now()}`.slice(0, 20);
}

export async function POST(req: Request) {
  try {
    const merchantID = process.env.ECPAY_MERCHANT_ID;
    const hashKey = process.env.ECPAY_HASH_KEY;
    const hashIV = process.env.ECPAY_HASH_IV;
    const stageUrl = process.env.ECPAY_STAGE_URL;

    if (!merchantID || !hashKey || !hashIV || !stageUrl) {
      return NextResponse.json(
        { error: "ECPay environment variables missing" },
        { status: 500 }
      );
    }

    const origin = new URL(req.url).origin;

    const params: Record<string, string> = {
      MerchantID: merchantID,
      MerchantTradeNo: createMerchantTradeNo(),
      MerchantTradeDate: getTradeDate(),
      PaymentType: "aio",
      TotalAmount: "5",
      TradeDesc: "金流測試",
      ItemName: "NT$1 金流測試商品",
      ReturnURL: `${origin}/api/payment/ecpay/notify`,
      OrderResultURL: `${origin}/payment/result`,
      ChoosePayment: "Credit",
      EncryptType: "1",
    };

    params.CheckMacValue = generateCheckMacValue(
      params,
      hashKey,
      hashIV
    );

    return NextResponse.json({
      action: stageUrl,
      params,
    });
  } catch (error) {
    console.error(error);

    return NextResponse.json(
      { error: "建立綠界付款失敗" },
      { status: 500 }
    );
  }
}