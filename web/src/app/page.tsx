import Link from "next/link";
import {
  ArrowUpRight,
  BookHeart,
  BrainCircuit,
  HeartHandshake,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

export default function HomePage() {
  return (
    <main className="hero-grid mx-auto flex min-h-screen max-w-[1480px] flex-col justify-between gap-10 px-4 py-5 lg:px-6">
      <header className="glass dashboard-orb flex items-center justify-between rounded-[2rem] px-5 py-4">
        <div>
          <p className="theme-kicker">GenZ AI Therapist</p>
          <p className="mt-1 text-sm text-[var(--muted)]">
            Gen Z-coded emotional support, reflection, and resource guidance
            without pretending to be clinical care.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/auth"
            className="button-ghost inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition"
          >
            Sign in
          </Link>
          <Link
            href="/app/chat"
            className="button-forest inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition"
          >
            Start yapping
            <ArrowUpRight size={16} />
          </Link>
        </div>
      </header>

      <section className="grid gap-8 py-8 lg:grid-cols-[1.1fr_0.9fr] lg:py-14">
        <div className="space-y-8">
          <div className="theme-chip inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs uppercase tracking-[0.3em] text-[var(--muted)]">
            non-clinical gen z support
          </div>
          <div className="space-y-5">
            <h1 className="max-w-4xl font-display text-[4rem] leading-[0.9] tracking-[-0.05em] md:text-[5.8rem]">
              When your mind gets loud, make the room feel softer.
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-[var(--muted)]">
              Yap things out, track your vibe, journal when the thought is too
              layered, and keep only the memory you actually want. This is a
              non-clinical listener, not diagnosis or therapy, and it should
              feel like talking to something helpful instead of a robot in a
              blazer.
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {[
              {
                label: "yap it out",
                value: "main chat",
                detail: "For spirals, overthinking, venting, or making one tiny plan so the day stops eating you.",
              },
              {
                label: "clock the pattern",
                value: "vibe check",
                detail: "Track mood, energy, and stress without turning yourself into a productivity spreadsheet.",
              },
              {
                label: "keep the lore",
                value: "private memory",
                detail: "You decide what this app is allowed to remember about you and what needs to stay gone.",
              },
            ].map((item, index) => (
              <article
                key={item.label}
                className={`${index === 0 ? "panel-clay" : index === 1 ? "panel-moss" : "panel-plum"} dashboard-orb rounded-[1.7rem] p-5`}
              >
                <p className="text-[11px] uppercase tracking-[0.24em] text-[var(--muted)]">
                  {item.label}
                </p>
                <p className="mt-3 font-display text-3xl leading-none">{item.value}</p>
                <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{item.detail}</p>
              </article>
            ))}
          </div>
          <div className="flex flex-wrap gap-3">
            {[
              "I need to vent, bad.",
              "Help me figure out what I’m even feeling.",
              "Can we make a tiny plan so I stop flopping?",
              "I want to check in without spiraling.",
            ].map((prompt) => (
              <span
                key={prompt}
                className="theme-chip rounded-full px-4 py-2 text-sm text-[var(--ink)]"
              >
                {prompt}
              </span>
            ))}
          </div>
        </div>

        <div className="grid gap-4">
          <article className="panel-dark dashboard-orb rounded-[2rem] p-6 text-[var(--paper)]">
            <p className="theme-kicker text-[rgba(255,236,219,0.68)]">how this feels</p>
            <h2 className="mt-3 max-w-md font-display text-5xl leading-[0.95]">
              More listener than lecturer. More mirror than machine.
            </h2>
            <p className="mt-4 max-w-md text-sm leading-7 text-[rgba(255,235,220,0.82)]">
              The whole interface is meant to reduce activation: warm colors,
              clear routes, grounded copy, and enough contrast that the app
              feels awake without feeling loud.
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              {[
                ["moss", "grounding and steadiness"],
                ["ember", "action and momentum"],
                ["plum", "reflection and pattern"],
              ].map(([name, meaning]) => (
                <div key={name} className="rounded-[1.2rem] border border-[rgba(255,236,219,0.08)] bg-[rgba(255,255,255,0.05)] px-4 py-3">
                  <p className="text-sm font-semibold capitalize">{name}</p>
                  <p className="mt-1 text-xs leading-6 text-[rgba(255,235,220,0.72)]">
                    {meaning}
                  </p>
                </div>
              ))}
            </div>
          </article>
          {[
            {
              icon: HeartHandshake,
              title: "Start messy",
              body: "This app is built for the first honest sentence, not the curated version of your life.",
            },
            {
              icon: BookHeart,
              title: "Journal when the thought is too big",
              body: "Some feelings need a whole page, not a single bubble. Both belong in the same spot.",
            },
            {
              icon: BrainCircuit,
              title: "See the whole week, not just today's chaos",
              body: "Vibe checks and pattern tea help you notice the repeat plotlines before everything blurs together.",
            },
            {
              icon: ShieldCheck,
              title: "Support with actual boundaries",
              body: "This space stays non-clinical and still takes crisis language seriously with direct safety guidance.",
            },
          ].map(({ icon: Icon, title, body }, index) => (
            <article
              key={title}
              className={`${index % 2 === 0 ? "panel-moss" : "panel-clay"} dashboard-orb rounded-[1.8rem] p-5`}
            >
              <div className={`inline-flex rounded-full p-3 ${index % 2 === 0 ? "bg-[rgba(53,88,78,0.14)] text-[var(--forest)]" : "bg-[rgba(182,103,67,0.14)] text-[var(--clay)]"}`}>
                <Icon size={18} />
              </div>
              <p className="mt-4 text-lg font-semibold">{title}</p>
              <p className="mt-2 text-sm leading-7 text-[var(--muted)]">{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel-dark dashboard-orb mb-6 grid gap-4 rounded-[2.2rem] p-6 text-[var(--paper)] lg:grid-cols-[1fr_auto] lg:items-center">
        <div>
          <p className="theme-kicker text-[rgba(255,232,215,0.72)]">Safety note</p>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-[rgba(255,232,215,0.86)]">
            GenZ AI Therapist is for emotional support, grounding, and getting
            your thoughts untangled. It is not emergency care or a replacement
            for a real therapist. If you think you may act on thoughts of
            self-harm or you are in immediate danger, contact local emergency
            services or a crisis resource right away.
          </p>
        </div>
        <Link
          href="/app/chat"
          className="inline-flex items-center gap-2 rounded-full bg-[var(--paper)] px-4 py-2 text-sm font-semibold text-[var(--ink)] shadow-[0_14px_28px_rgba(20,12,8,0.18)] transition hover:-translate-y-0.5 hover:bg-white"
        >
          Open the chat
          <Sparkles size={16} />
        </Link>
      </section>
    </main>
  );
}
