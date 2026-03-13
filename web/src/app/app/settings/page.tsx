import { MemoryControls } from "@/components/memory-controls";
import { getMemoryItems } from "@/lib/memory";
import { getViewer } from "@/lib/viewer";

export default async function SettingsPage() {
  const { user } = await getViewer();
  const userId = user?.id;

  const memory = await getMemoryItems(userId);

  return (
    <div className="app-scrollbar grid h-full min-h-0 gap-6 overflow-y-auto pr-1 xl:overflow-hidden xl:pr-0 xl:grid-cols-2 xl:grid-rows-[minmax(0,0.88fr)_minmax(0,1.12fr)]">
      <section className="panel-moss dashboard-orb flex min-h-0 flex-col rounded-[2rem] p-6">
        <p className="theme-kicker">your space</p>
        <h1 className="mt-2 font-display text-4xl leading-none">Privacy, pace, and no weirdness.</h1>
        <div className="app-scrollbar mt-6 min-h-0 flex-1 space-y-4 overflow-y-auto pr-1 text-sm leading-7 text-[var(--muted)]">
          <p>
            This space should feel chill, not invasive. Keep what helps, hide what does not, and let the app remember only what actually feels useful.
          </p>
          <div className="panel-plum rounded-[1.5rem] p-5">
            <p className="font-semibold text-[var(--ink)]">what this is for</p>
            <ul className="mt-3 space-y-2">
              <li>Keep or hide memory bits that shape future chats.</li>
              <li>Make sure your support space feels personal, not invasive.</li>
              <li>Come back here anytime the app starts remembering the wrong lore.</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="panel-clay dashboard-orb flex min-h-0 flex-col rounded-[2rem] p-6">
        <p className="theme-kicker">care style</p>
        <div className="app-scrollbar mt-5 min-h-0 flex-1 space-y-4 overflow-y-auto pr-1 text-sm leading-7 text-[var(--muted)]">
          <p>
            The tone of this app should stay warm, direct, and very not fake. It is here to help you slow down, not sell you productivity theater.
          </p>
          <div className="panel-dark rounded-[1.4rem] p-4 text-[var(--paper)]">
            <p className="font-semibold text-[var(--paper)]">boundaries</p>
            <p className="mt-2 text-[rgba(255,232,215,0.82)]">
              This app is for emotional support, reflection, and grounding. It is not emergency care and it is not pretending to be a licensed clinician.
            </p>
          </div>
          <div className="panel-plum rounded-[1.4rem] p-4">
            <p className="font-semibold text-[var(--ink)]">account</p>
            <p className="mt-2">
              {userId
                ? "Your chats can follow you between visits when you sign in."
                : "Sign-in is still getting wired for cross-device persistence."}
            </p>
          </div>
        </div>
      </section>

      <div className="min-h-0 xl:col-span-2">
        <MemoryControls initialItems={memory.items} demoMode={memory.demoMode} />
      </div>
    </div>
  );
}
