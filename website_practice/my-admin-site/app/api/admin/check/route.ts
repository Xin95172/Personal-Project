import { createClient } from "@/lib/supabase/server";

export async function GET() {
  const supabase = await createClient();

  // 檢查瀏覽器目前登入的是誰
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  if (userError || !user) {
    return Response.json(
      {
        success: false,
        error: "目前沒有登入",
      },
      {
        status: 401,
      },
    );
  }

  // 執行我們在資料庫建立的 is_admin()
  const {
    data: isAdmin,
    error: adminError,
  } = await supabase.rpc("is_admin");

  if (adminError) {
    return Response.json(
      {
        success: false,
        error: adminError.message,
      },
      {
        status: 500,
      },
    );
  }

  return Response.json({
    success: true,
    email: user.email,
    isAdmin,
  });
}