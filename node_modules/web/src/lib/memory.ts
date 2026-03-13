import { unstable_noStore as noStore } from "next/cache";

import { demoMemoryItems } from "@/lib/demo-data";
import { createServerSupabase } from "@/lib/supabase/server";

export type MemoryItemView = {
  id: string;
  content: string;
  category: string;
  status: "pending" | "approved" | "hidden";
  createdAt: string;
};

type MemoryItemRow = {
  id: string;
  content: string;
  category: string | null;
  status: "pending" | "approved" | "hidden";
  created_at: string;
};

export async function getMemoryItems(userId?: string) {
  noStore();

  if (!userId) {
    return {
      items: demoMemoryItems satisfies MemoryItemView[],
      demoMode: true,
    };
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return {
      items: [] satisfies MemoryItemView[],
      demoMode: false,
    };
  }

  try {
    const { data } = await supabase
      .from("memory_items")
      .select("id,content,category,status,created_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(20);

    return {
      items: ((data ?? []) as MemoryItemRow[]).map((item) => ({
        id: item.id,
        content: item.content,
        category: item.category ?? "general",
        status: item.status,
        createdAt: new Date(item.created_at).toLocaleString(),
      })),
      demoMode: false,
    };
  } catch {
    return {
      items: [] satisfies MemoryItemView[],
      demoMode: false,
    };
  }
}
