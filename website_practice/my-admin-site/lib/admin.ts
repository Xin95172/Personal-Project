import { createClient } from "@/lib/supabase/server";

export async function requireAdmin() {
  const supabase = await createClient();
  const { data: { user }, error: userError } = await supabase.auth.getUser();

  if (userError || !user) {
    return { supabase, error: "請先登入", status: 401 } as const;
  }

  const { data: isAdmin, error: adminError } = await supabase.rpc("is_admin");
  if (adminError) {
    return { supabase, error: adminError.message, status: 500 } as const;
  }
  if (!isAdmin) {
    return { supabase, error: "沒有管理員權限", status: 403 } as const;
  }

  return { supabase, user, error: null, status: null } as const;
}
