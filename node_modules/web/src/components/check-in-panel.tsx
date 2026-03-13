"use client";

import { useState, useTransition } from "react";

type CheckInPanelProps = {
  initialMood: number;
  initialEnergy: number;
  initialStress: number;
  demoMode: boolean;
  initialNote?: string;
};

export function CheckInPanel({
  initialMood,
  initialEnergy,
  initialStress,
  demoMode,
  initialNote = "",
}: CheckInPanelProps) {
  const [mood, setMood] = useState(initialMood);
  const [energy, setEnergy] = useState(initialEnergy);
  const [stress, setStress] = useState(initialStress);
  const [note, setNote] = useState(initialNote);
  const [status, setStatus] = useState(
    demoMode ? "You can still do a vibe check while sign-in finishes cooking." : "A 30-second vibe check is enough.",
  );
  const [isPending, startTransition] = useTransition();

  const metrics = [
    { label: "Mood", value: mood, setValue: setMood },
    { label: "Energy", value: energy, setValue: setEnergy },
    { label: "Stress", value: stress, setValue: setStress },
  ];

  const saveCheckIn = () => {
    startTransition(async () => {
      const response = await fetch("/api/check-in", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mood, energy, stress, note }),
      });

      const payload = (await response.json()) as { ok?: boolean };
      if (!response.ok || !payload.ok) {
        setStatus("Could not save check-in.");
        return;
      }

      setStatus(demoMode ? "Saved for this session." : "Check-in saved.");
    });
  };

  return (
    <section className="app-scrollbar panel-moss dashboard-orb h-full overflow-y-auto rounded-[2rem] p-6">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[rgba(91,58,38,0.08)] pb-5">
        <div>
          <p className="theme-kicker">Daily vibe check</p>
          <h1 className="mt-2 font-display text-4xl leading-none">So... what is the vibe today?</h1>
          <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--muted)]">
            Do not overthink the numbers. They are just handles, not grades.
          </p>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-3">
        {[
          "heavy but functioning",
          "low energy, high pressure",
          "honestly kinda okay today",
        ].map((line) => (
          <button
            key={line}
            type="button"
            onClick={() => setNote(line)}
            className="theme-chip rounded-full px-4 py-2 text-sm transition"
          >
            {line}
          </button>
        ))}
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {metrics.map((metric) => (
          <label
            key={metric.label}
            className={`${metric.label === "Stress" ? "panel-clay" : metric.label === "Energy" ? "panel-plum" : "panel-moss"} rounded-[1.6rem] p-5`}
          >
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm text-[var(--muted)]">{metric.label}</p>
              <p className="font-semibold">{metric.value}</p>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={metric.value}
              onChange={(event) => metric.setValue(Number(event.target.value))}
              className="mt-4 w-full accent-[var(--forest)]"
            />
          </label>
        ))}
      </div>

      <div className="panel-dark dashboard-orb mt-6 rounded-[1.6rem] p-5 text-[var(--paper)]">
        <p className="text-sm font-semibold">Quick note</p>
        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="drop the honest one-paragraph vibe recap"
          className="mt-3 min-h-28 w-full resize-none bg-transparent text-sm leading-7 text-[rgba(255,232,215,0.88)] outline-none placeholder:text-[rgba(255,232,215,0.54)]"
        />
      </div>

      <div className="mt-5 flex items-center justify-between gap-4">
        <p className="text-sm text-[var(--muted)]">{status}</p>
        <button
          type="button"
          onClick={saveCheckIn}
          disabled={isPending}
          className="button-forest rounded-full px-4 py-2 text-sm font-medium transition disabled:opacity-60"
        >
          {isPending ? "Saving..." : "save vibe check"}
        </button>
      </div>
    </section>
  );
}
