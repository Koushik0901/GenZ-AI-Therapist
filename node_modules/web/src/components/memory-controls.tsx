"use client";

import { useState, useTransition } from "react";

import type { MemoryItemView } from "@/lib/memory";

type MemoryControlsProps = {
  initialItems: MemoryItemView[];
  demoMode: boolean;
};

type MemoryStatus = MemoryItemView["status"];

const statusTone: Record<MemoryStatus, string> = {
  approved: "bg-[rgba(53,88,78,0.12)] text-[var(--forest)]",
  pending: "bg-[rgba(224,163,90,0.16)] text-[#9c6216]",
  hidden: "bg-[rgba(91,58,38,0.12)] text-[var(--muted)]",
};

export function MemoryControls({ initialItems, demoMode }: MemoryControlsProps) {
  const [items, setItems] = useState(initialItems);
  const [status, setStatus] = useState(
    demoMode
      ? "You can still shape the memory vibe here while signed-out."
      : "Only keep the bits that would actually help future convos feel more known, not weird.",
  );
  const [isPending, startTransition] = useTransition();

  const updateStatus = (id: string, nextStatus: MemoryStatus) => {
    startTransition(async () => {
      if (demoMode) {
        setItems((current) =>
          current.map((item) => (item.id === id ? { ...item, status: nextStatus } : item)),
        );
        setStatus(`Memory preference updated to ${nextStatus}.`);
        return;
      }

      const response = await fetch(`/api/memory/${id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ status: nextStatus }),
      });

      if (!response.ok) {
        setStatus("Could not update memory item.");
        return;
      }

      setItems((current) =>
        current.map((item) => (item.id === id ? { ...item, status: nextStatus } : item)),
      );
      setStatus(`Memory preference updated to ${nextStatus}.`);
    });
  };

  return (
    <section className="panel-plum dashboard-orb flex h-full min-h-0 flex-col rounded-[2rem] p-6">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[rgba(91,58,38,0.08)] pb-5">
        <div>
          <p className="theme-kicker">lore controls</p>
          <h2 className="mt-2 font-display text-4xl leading-none">Choose what stays.</h2>
        </div>
        <div className="rounded-full bg-[rgba(95,102,114,0.12)] px-3 py-1 text-xs font-semibold text-[var(--plum-deep)]">
          {demoMode ? "try it out" : "private to your account"}
        </div>
      </div>

      <p className="mt-5 text-sm leading-7 text-[var(--muted)]">{status}</p>

      <div className="app-scrollbar mt-5 min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
        {items.length ? (
          items.map((item, index) => (
            <article
              key={item.id}
              className={`${index % 2 === 0 ? "panel-clay" : "panel-moss"} rounded-[1.5rem] p-5`}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="space-y-2">
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${statusTone[item.status]}`}
                  >
                    {item.status}
                  </span>
                  <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                    {item.category} · {item.createdAt}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={isPending}
                    onClick={() => updateStatus(item.id, "approved")}
                    className="button-forest rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] disabled:opacity-60"
                  >
                    keep
                  </button>
                  <button
                    type="button"
                    disabled={isPending}
                    onClick={() => updateStatus(item.id, "hidden")}
                    className="button-ghost rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--ink)] disabled:opacity-60"
                  >
                    hide
                  </button>
                  <button
                    type="button"
                    disabled={isPending}
                    onClick={() => updateStatus(item.id, "pending")}
                    className="rounded-full bg-[rgba(224,163,90,0.16)] px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#9c6216] disabled:opacity-60"
                  >
                    reset
                  </button>
                </div>
              </div>
              <p className="mt-4 text-sm leading-7 text-[var(--ink)]">{item.content}</p>
            </article>
          ))
        ) : (
          <div className="rounded-[1.4rem] border border-dashed border-[rgba(91,58,38,0.18)] bg-white/55 p-5 text-sm leading-7 text-[var(--muted)]">
            Nothing is saved to memory yet, which is honestly fine. Keep it lightweight until remembering actually feels useful.
          </div>
        )}
      </div>
    </section>
  );
}
