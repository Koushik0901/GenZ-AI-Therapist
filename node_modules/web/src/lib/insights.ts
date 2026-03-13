import { unstable_noStore as noStore } from "next/cache";

import { checkInSnapshot, insightCards } from "@/lib/demo-data";
import { createServerSupabase } from "@/lib/supabase/server";

type SignalTrendRow = {
  mood_score: number;
  energy_score: number;
  stress_score: number;
  created_at: string;
  source: "manual" | "chat";
};

export type InsightCard = {
  label: string;
  value: string;
  detail: string;
};

export type InsightTrendPoint = {
  date: string;
  label: string;
  mood: number;
  energy: number;
  stress: number;
};

export type InsightWeeklyPoint = {
  label: string;
  mood: number;
  energy: number;
  stress: number;
};

export type InsightsPayload = {
  checkInSnapshot: {
    mood: number;
    energy: number;
    stress: number;
    streak: number;
  };
  insightCards: InsightCard[];
  trendSeries: InsightTrendPoint[];
  weeklySeries: InsightWeeklyPoint[];
  demoMode: boolean;
};

type MessageSignalRow = {
  created_at: string;
  model_metadata:
    | {
        conversation_wellness?: {
          mood?: number;
          energy?: number;
          stress?: number;
        };
      }
    | null;
};

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }

  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function computeStreak(checkIns: SignalTrendRow[]) {
  const uniqueDays = [...new Set(checkIns.map((entry) => entry.created_at.slice(0, 10)))];
  if (!uniqueDays.length) {
    return 0;
  }

  let streak = 1;
  const cursor = new Date(`${uniqueDays[0]}T00:00:00Z`);

  for (let index = 1; index < uniqueDays.length; index += 1) {
    cursor.setUTCDate(cursor.getUTCDate() - 1);
    const expectedDay = cursor.toISOString().slice(0, 10);
    if (uniqueDays[index] !== expectedDay) {
      break;
    }
    streak += 1;
  }

  return streak;
}

function deriveTrendLabel(recentMoodAverage: number, previousMoodAverage: number) {
  if (!previousMoodAverage) {
    return "Calibrating";
  }

  const delta = recentMoodAverage - previousMoodAverage;
  if (delta >= 8) {
    return "Lifting";
  }
  if (delta <= -8) {
    return "Rougher";
  }

  return "Steadier";
}

function derivePressureLabel(energyAverage: number, stressAverage: number) {
  if (stressAverage - energyAverage >= 18) {
    return "Overloaded";
  }
  if (energyAverage - stressAverage >= 15) {
    return "Resourced";
  }

  return "Mixed";
}

function buildInsightCards(args: {
  checkIns: SignalTrendRow[];
  journalCount: number;
  sessionCount: number;
  approvedMemoryCount: number;
}) {
  const { checkIns, journalCount, sessionCount, approvedMemoryCount } = args;
  const recentWindow = checkIns.slice(0, 3);
  const previousWindow = checkIns.slice(3, 6);
  const recentMoodAverage = average(recentWindow.map((entry) => entry.mood_score));
  const previousMoodAverage = average(previousWindow.map((entry) => entry.mood_score));
  const moodTrend = deriveTrendLabel(recentMoodAverage, previousMoodAverage);
  const energyAverage = average(checkIns.slice(0, 7).map((entry) => entry.energy_score));
  const stressAverage = average(checkIns.slice(0, 7).map((entry) => entry.stress_score));
  const streak = computeStreak(checkIns);

  const cards: InsightCard[] = [
    {
      label: "Mood trend",
      value: moodTrend,
      detail: recentWindow.length
        ? `Average mood is ${recentMoodAverage}/100 across your latest ${recentWindow.length} logged signals.`
        : "Give it a few chats or vibe checks and the trend signal will stop being a guess.",
    },
    {
      label: "Return rhythm",
      value: streak ? `${streak}-day streak` : "Just starting",
      detail: `${sessionCount} saved sessions and ${journalCount} journal entries are part of the current support loop.`,
    },
    {
      label: "Pressure balance",
      value: derivePressureLabel(energyAverage, stressAverage),
      detail: `Recent averages are ${energyAverage}/100 energy and ${stressAverage}/100 stress.`,
    },
    {
      label: "Memory runway",
      value: approvedMemoryCount ? `${approvedMemoryCount} approved` : "No memory saved",
      detail: approvedMemoryCount
        ? "Approved memories are ready to personalize future replies once the memory layer is connected."
        : "Nothing approved yet, which keeps the system lightweight and private by default.",
    },
  ];

  return cards;
}

function formatShortDate(value: string) {
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function startOfUtcWeek(dateString: string) {
  const date = new Date(dateString);
  const day = date.getUTCDay();
  const diff = day === 0 ? -6 : 1 - day;
  date.setUTCDate(date.getUTCDate() + diff);
  date.setUTCHours(0, 0, 0, 0);
  return date;
}

function buildTrendSeries(checkIns: SignalTrendRow[]) {
  return [...checkIns]
    .reverse()
    .map((entry) => ({
      date: entry.created_at,
      label: formatShortDate(entry.created_at),
      mood: entry.mood_score,
      energy: entry.energy_score,
      stress: entry.stress_score,
    }));
}

function buildWeeklySeries(checkIns: SignalTrendRow[]) {
  const buckets = new Map<
    string,
    { label: string; mood: number[]; energy: number[]; stress: number[] }
  >();

  for (const entry of checkIns) {
    const weekStart = startOfUtcWeek(entry.created_at);
    const key = weekStart.toISOString().slice(0, 10);
    const existing = buckets.get(key) ?? {
      label: weekStart.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      }),
      mood: [],
      energy: [],
      stress: [],
    };

    existing.mood.push(entry.mood_score);
    existing.energy.push(entry.energy_score);
    existing.stress.push(entry.stress_score);
    buckets.set(key, existing);
  }

  return [...buckets.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .slice(-4)
    .map(([, bucket]) => ({
      label: bucket.label,
      mood: average(bucket.mood),
      energy: average(bucket.energy),
      stress: average(bucket.stress),
    }));
}

function extractConversationSignals(rows: MessageSignalRow[]) {
  return rows
    .map((row) => {
      const signal = row.model_metadata?.conversation_wellness;
      if (
        typeof signal?.mood !== "number" ||
        typeof signal?.energy !== "number" ||
        typeof signal?.stress !== "number"
      ) {
        return null;
      }

      return {
        mood_score: signal.mood,
        energy_score: signal.energy,
        stress_score: signal.stress,
        created_at: row.created_at,
        source: "chat" as const,
      };
    })
    .filter(
      (
        row,
      ): row is {
        mood_score: number;
        energy_score: number;
        stress_score: number;
        created_at: string;
        source: "chat";
      } => row !== null,
    );
}

export async function getInsights(userId?: string) {
  noStore();

  if (!userId) {
    return {
      checkInSnapshot,
      insightCards,
      trendSeries: [
        { date: "demo-1", label: "Mon", mood: 56, energy: 42, stress: 71 },
        { date: "demo-2", label: "Tue", mood: 60, energy: 44, stress: 66 },
        { date: "demo-3", label: "Wed", mood: 62, energy: 40, stress: 74 },
        { date: "demo-4", label: "Thu", mood: 66, energy: 47, stress: 61 },
        { date: "demo-5", label: "Fri", mood: 69, energy: 52, stress: 58 },
      ],
      weeklySeries: [
        { label: "Feb 17", mood: 55, energy: 41, stress: 72 },
        { label: "Feb 24", mood: 58, energy: 43, stress: 68 },
        { label: "Mar 3", mood: 61, energy: 46, stress: 64 },
        { label: "Mar 10", mood: 66, energy: 49, stress: 59 },
      ],
      demoMode: true,
    } satisfies InsightsPayload;
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return {
      checkInSnapshot: {
        mood: 0,
        energy: 0,
        stress: 0,
        streak: 0,
      },
      insightCards: buildInsightCards({
        checkIns: [],
        journalCount: 0,
        sessionCount: 0,
        approvedMemoryCount: 0,
      }),
      trendSeries: [],
      weeklySeries: [],
      demoMode: false,
    } satisfies InsightsPayload;
  }

  try {
    const [
      { data: checkIns },
      { data: messageSignals },
      { count: sessionCount },
      { count: journalCount },
      { count: approvedMemoryCount },
    ] = await Promise.all([
      supabase
        .from("daily_checkins")
        .select("mood_score,energy_score,stress_score,created_at")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(14),
      supabase
        .from("messages")
        .select("created_at,model_metadata")
        .eq("role", "assistant")
        .order("created_at", { ascending: false })
        .limit(24),
      supabase
        .from("chat_sessions")
        .select("id", { count: "exact", head: true })
        .eq("user_id", userId)
        .eq("archived", false),
      supabase
        .from("journal_entries")
        .select("id", { count: "exact", head: true })
        .eq("user_id", userId),
      supabase
        .from("memory_items")
        .select("id", { count: "exact", head: true })
        .eq("user_id", userId)
        .eq("status", "approved"),
    ]);

    const manualRows = ((checkIns ?? []) as Array<{
      mood_score: number;
      energy_score: number;
      stress_score: number;
      created_at: string;
    }>).map((entry) => ({
      ...entry,
      source: "manual" as const,
    }));
    const inferredRows = extractConversationSignals(
      (messageSignals ?? []) as MessageSignalRow[],
    );
    const trendRows = [...manualRows, ...inferredRows]
      .sort((left, right) => right.created_at.localeCompare(left.created_at))
      .slice(0, 24);
    const latest = trendRows[0];

    return {
      checkInSnapshot: latest
        ? {
            mood: latest.mood_score,
            energy: latest.energy_score,
            stress: latest.stress_score,
            streak: computeStreak(trendRows),
          }
        : {
            ...checkInSnapshot,
            streak: 0,
          },
      insightCards: buildInsightCards({
        checkIns: trendRows,
        journalCount: journalCount ?? 0,
        sessionCount: sessionCount ?? 0,
        approvedMemoryCount: approvedMemoryCount ?? 0,
      }),
      trendSeries: buildTrendSeries(trendRows),
      weeklySeries: buildWeeklySeries(trendRows),
      demoMode: false,
    } satisfies InsightsPayload;
  } catch {
    return {
      checkInSnapshot: {
        mood: 0,
        energy: 0,
        stress: 0,
        streak: 0,
      },
      insightCards: buildInsightCards({
        checkIns: [],
        journalCount: 0,
        sessionCount: 0,
        approvedMemoryCount: 0,
      }),
      trendSeries: [],
      weeklySeries: [],
      demoMode: false,
    } satisfies InsightsPayload;
  }
}
