import { unstable_noStore as noStore } from "next/cache";

import { checkInSnapshot, journalEntries } from "@/lib/demo-data";
import { createServerSupabase } from "@/lib/supabase/server";

type JournalEntryRow = {
  id: string;
  title: string;
  body: string;
  mood: string | null;
  created_at: string;
};

type CheckInRow = {
  id: string;
  mood_score: number;
  energy_score: number;
  stress_score: number;
  note: string | null;
  created_at: string;
};

export async function getJournalEntries(userId?: string) {
  noStore();

  if (!userId) {
    return journalEntries;
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return [];
  }

  try {
    const { data } = await supabase
      .from("journal_entries")
      .select("id,title,body,mood,created_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(12);

    if (!data?.length) {
      return [];
    }

    return (data as JournalEntryRow[]).map((entry) => ({
      id: entry.id,
      title: entry.title,
      body: entry.body,
      mood: entry.mood ?? "Unlabeled",
      createdAt: new Date(entry.created_at).toLocaleString(),
    }));
  } catch {
    return [];
  }
}

export async function getLatestCheckIn(userId?: string) {
  noStore();

  if (!userId) {
    return {
      mood: checkInSnapshot.mood,
      energy: checkInSnapshot.energy,
      stress: checkInSnapshot.stress,
      note: "",
    };
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return {
      mood: 50,
      energy: 50,
      stress: 50,
      note: "",
    };
  }

  try {
    const { data } = await supabase
      .from("daily_checkins")
      .select("id,mood_score,energy_score,stress_score,note,created_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    if (!data) {
      return {
        mood: checkInSnapshot.mood,
        energy: checkInSnapshot.energy,
        stress: checkInSnapshot.stress,
        note: "",
      };
    }

    const row = data as CheckInRow;
    return {
      mood: row.mood_score,
      energy: row.energy_score,
      stress: row.stress_score,
      note: row.note ?? "",
    };
  } catch {
    return {
      mood: 50,
      energy: 50,
      stress: 50,
      note: "",
    };
  }
}
