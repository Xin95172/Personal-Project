export async function POST(request: Request) {
  const formData = await request.formData();

  const data = Object.fromEntries(formData.entries());

  console.log("ECPay notify:", data);

  // 綠界要求付款結果通知成功接收後回應 1|OK
  return new Response("1|OK", {
    status: 200,
    headers: {
      "Content-Type": "text/plain",
    },
  });
}