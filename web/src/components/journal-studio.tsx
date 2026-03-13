"use client";

import { useState, useTransition } from "react";

type JournalEntry = {
  id: string;
  title: string;
  mood: string;
  createdAt: string;
  body: string;
};

type JournalStudioProps = {
  initialEntries: JournalEntry[];
  demoMode: boolean;
};

export function JournalStudio({ initialEntries, demoMode }: JournalStudioProps) {
  const [entries, setEntries] = useState(initialEntries);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [mood, setMood] = useState("Tender");
  const [status, setStatus] = useState(
    demoMode ? "You can still write here while sign-in finishes cooking." : "Drop the longer version here.",
  );
  const [isPending, startTransition] = useTransition();

  const saveEntry = () => {
    if (!title.trim() || !body.trim()) {
      setStatus("Title and journal text are required.");
      return;
    }

    startTransition(async () => {
      const response = await fetch("/api/journal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, body, mood }),
      });

      const payload = (await response.json()) as {
        ok?: boolean;
        error?: { formErrors?: string[] };
        payload?: { title: string; body: string; mood?: string };
      };

      if (!response.ok || !payload.ok) {
        setStatus("Could not save journal entry.");
        return;
      }

      setEntries((current) => [
        {
          id: crypto.randomUUID(),
          title: payload.payload?.title ?? title,
          body: payload.payload?.body ?? body,
          mood: payload.payload?.mood ?? mood,
          createdAt: "Just now",
        },
        ...current,
      ]);
      setTitle("");
      setBody("");
      setMood("Tender");
      setStatus(demoMode ? "Saved for this session." : "Journal entry saved.");
    });
  };

  return (
    <div className="grid h-full min-h-0 gap-6 overflow-hidden xl:grid-cols-[1.05fr_0.95fr]">
      <section className="panel-plum dashboard-orb flex min-h-0 flex-col rounded-[2rem] p-6">
        <p className="theme-kicker">Journal studio</p>
        <h1 className="mt-2 font-display text-4xl leading-none">Turn the spiral into actual words.</h1>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--muted)]">
          Use this when the thought is too layered for chat. You do not need to sound deep. Just get closer to what is actually true.
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          {[
            "what am I carrying that nobody can see?",
            "what triggered me way more than it should have?",
            "what do I need instead of what I keep forcing?",
          ].map((prompt) => (
            <button
              key={prompt}
              type="button"
              onClick={() => setBody(prompt)}
              className="theme-chip rounded-full px-4 py-2 text-sm transition"
            >
              {prompt}
            </button>
          ))}
        </div>
        <div className="app-scrollbar mt-6 min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
          <input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder="name the lore drop"
            className="theme-input w-full rounded-[1.2rem] px-4 py-3 text-sm"
          />
          <select
            value={mood}
            onChange={(event) => setMood(event.target.value)}
            className="theme-input w-full rounded-[1.2rem] px-4 py-3 text-sm"
          >
            {["Tender", "Wired", "Flat", "Steady", "Raw"].map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
          <textarea
            value={body}
            onChange={(event) => setBody(event.target.value)}
            placeholder="What is true right now, even if it is messy?"
            className="theme-input min-h-80 w-full resize-none rounded-[1.5rem] px-4 py-4 text-sm leading-7"
          />
          <div className="flex items-center justify-between gap-4">
            <p className="text-sm text-[var(--muted)]">{status}</p>
            <button
              type="button"
              onClick={saveEntry}
              disabled={isPending}
              className="button-coral rounded-full px-4 py-2 text-sm font-medium transition disabled:opacity-60"
            >
              {isPending ? "Saving..." : "save it"}
            </button>
          </div>
        </div>
      </section>

      <section className="app-scrollbar min-h-0 space-y-4 overflow-y-auto pr-1">
        <div className="panel-dark dashboard-orb rounded-[1.8rem] p-5 text-[var(--paper)]">
          <p className="theme-kicker text-[rgba(255,232,215,0.72)]">why dump it here</p>
          <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
            <span className="text-[rgba(255,232,215,0.84)]">
              Chat is for back-and-forth. Journaling is for when your brain needs room to fully yap without interruption.
            </span>
          </p>
        </div>
        {entries.map((entry, index) => (
          <article key={entry.id} className={`${index % 2 === 0 ? "panel-clay" : "panel-moss"} dashboard-orb rounded-[1.8rem] p-5`}>
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="font-semibold">{entry.title}</p>
                <p className="mt-1 text-xs uppercase tracking-[0.22em] text-[var(--muted)]">
                  {entry.createdAt}
                </p>
              </div>
              <span className="rounded-full bg-[rgba(95,102,114,0.12)] px-3 py-1 text-xs font-semibold text-[var(--plum-deep)]">
                {entry.mood}
              </span>
            </div>
            <p className="mt-4 text-sm leading-7 text-[var(--muted)]">{entry.body}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
