"use client";

import { useEffect, useState } from "react";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { InsightsPayload } from "@/lib/insights";

type InsightsDashboardProps = {
  insights: InsightsPayload;
};

export function InsightsDashboard({ insights }: InsightsDashboardProps) {
  const [chartsReady, setChartsReady] = useState(false);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      setChartsReady(true);
    });

    return () => window.cancelAnimationFrame(frame);
  }, []);

  const metrics = [
    ["Mood", insights.checkInSnapshot.mood],
    ["Energy", insights.checkInSnapshot.energy],
    ["Stress", insights.checkInSnapshot.stress],
    ["Streak", insights.checkInSnapshot.streak],
  ];

  return (
    <div className="app-scrollbar h-full space-y-6 overflow-y-auto pr-1">
      <section className="panel-plum dashboard-orb rounded-[2rem] p-6">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[rgba(91,58,38,0.08)] pb-5">
          <div>
            <p className="theme-kicker">pattern tea</p>
            <h1 className="mt-2 font-display text-4xl leading-none">
              The tea is in the pattern.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--muted)]">
              This page is here to make your emotional pattern easier to spot,
              not to rank your worth. GenZ AI Therapist is still non-clinical,
              more listener than therapist, and better at surfacing patterns
              than replacing real care.
            </p>
          </div>
          <div className="rounded-full bg-[rgba(95,102,114,0.12)] px-3 py-1 text-xs font-semibold text-[var(--plum-deep)]">
            {insights.demoMode
              ? "sample view"
              : "built from your recent vibe checks"}
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-4">
          {metrics.map(([label, value], index) => (
            <div
              key={label}
              className={`${index === 0 ? "panel-moss" : index === 1 ? "panel-clay" : index === 2 ? "panel-dark text-[var(--paper)]" : "glass"} dashboard-orb rounded-[1.6rem] p-5`}
            >
              <p className={`text-sm ${index === 2 ? "text-[rgba(255,232,215,0.74)]" : "text-[var(--muted)]"}`}>{label}</p>
              <p className={`mt-3 font-display text-5xl leading-none ${index === 2 ? "text-[var(--paper)]" : ""}`}>{value}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,0.95fr)]">
        <section className="panel-moss dashboard-orb min-w-0 rounded-[1.9rem] p-6">
          <p className="theme-kicker">trend view</p>
          <h2 className="mt-2 font-display text-3xl leading-none">
            Mood, energy, and stress over time
          </h2>
          <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
            Higher mood and energy usually feel better. Higher stress usually
            means more pressure.
          </p>
          <div className="mt-6 h-[320px] min-w-0">
            {chartsReady && insights.trendSeries.length ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={280} minHeight={320}>
                <AreaChart data={insights.trendSeries}>
                  <defs>
                    <linearGradient id="moodFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4d8a73" stopOpacity={0.28} />
                      <stop offset="95%" stopColor="#4d8a73" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="energyFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#d49a58" stopOpacity={0.22} />
                      <stop offset="95%" stopColor="#d49a58" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="stressFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#d46a54" stopOpacity={0.18} />
                      <stop offset="95%" stopColor="#d46a54" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(91,58,38,0.08)" vertical={false} />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 100]}
                    width={36}
                  />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="mood"
                    stroke="#4d8a73"
                    fill="url(#moodFill)"
                    strokeWidth={2.5}
                  />
                  <Area
                    type="monotone"
                    dataKey="energy"
                    stroke="#d49a58"
                    fill="url(#energyFill)"
                    strokeWidth={2.5}
                  />
                  <Area
                    type="monotone"
                    dataKey="stress"
                    stroke="#d46a54"
                    fill="url(#stressFill)"
                    strokeWidth={2.5}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center rounded-[1.4rem] border border-dashed border-[rgba(91,58,38,0.16)] bg-white/58 px-6 text-center text-sm leading-7 text-[var(--muted)]">
                {chartsReady
                  ? "No graph yet. A few vibe checks will turn this into something useful fast."
                  : "Loading the pattern view..."}
              </div>
            )}
          </div>
        </section>

        <section className="panel-clay dashboard-orb min-w-0 rounded-[1.9rem] p-6">
          <p className="theme-kicker">weekly averages</p>
          <h2 className="mt-2 font-display text-3xl leading-none">
            Week by week
          </h2>
          <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
            This smooths out random bad days so you can clock the broader arc.
          </p>
          <div className="mt-6 h-[320px] min-w-0">
            {chartsReady && insights.weeklySeries.length ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={280} minHeight={320}>
                <BarChart data={insights.weeklySeries} barGap={10}>
                  <CartesianGrid stroke="rgba(91,58,38,0.08)" vertical={false} />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 100]}
                    width={36}
                  />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="mood" fill="#4d8a73" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="energy" fill="#d49a58" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="stress" fill="#d46a54" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center rounded-[1.4rem] border border-dashed border-[rgba(91,58,38,0.16)] bg-white/58 px-6 text-center text-sm leading-7 text-[var(--muted)]">
                {chartsReady
                  ? "Once you have at least a few check-ins, this view starts making the week-to-week shifts way easier to read."
                  : "Loading the weekly view..."}
              </div>
            )}
          </div>
        </section>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {insights.insightCards.map((card, index) => (
          <article
            key={card.label}
            className={`${index % 3 === 0 ? "panel-dark text-[var(--paper)]" : index % 3 === 1 ? "panel-moss" : "panel-plum"} dashboard-orb rounded-[1.8rem] p-6`}
          >
            <p className={`text-[11px] uppercase tracking-[0.28em] ${index % 3 === 0 ? "text-[rgba(255,232,215,0.72)]" : "text-[var(--muted)]"}`}>
              {card.label}
            </p>
            <p className="mt-3 font-display text-4xl leading-none">{card.value}</p>
            <p className={`mt-3 text-sm leading-7 ${index % 3 === 0 ? "text-[rgba(255,232,215,0.82)]" : "text-[var(--muted)]"}`}>
              {card.detail}
            </p>
          </article>
        ))}
      </div>
    </div>
  );
}
