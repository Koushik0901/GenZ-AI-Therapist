function PulseBlock({ className }: { className: string }) {
  return <div className={`animate-pulse rounded-[1.2rem] bg-[rgba(91,58,38,0.08)] ${className}`} />;
}

export function AppRouteLoading() {
  return (
    <div className="grid h-[100dvh] max-w-[1600px] grid-cols-[78px_minmax(0,1fr)] gap-4 overflow-hidden px-4 py-4 lg:px-6">
      <aside className="glass flex h-full min-h-0 flex-col items-center justify-between rounded-[2rem] px-3 py-4 shadow-[0_22px_60px_rgba(55,31,19,0.12)]">
        <div className="flex flex-col items-center gap-3">
          <PulseBlock className="h-12 w-12 rounded-[1.3rem]" />
          <div className="h-px w-8 bg-[rgba(91,58,38,0.12)]" />
          <div className="flex flex-col gap-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <PulseBlock key={index} className="h-12 w-12 rounded-[1.1rem]" />
            ))}
          </div>
        </div>
        <PulseBlock className="h-12 w-12 rounded-[1.1rem]" />
      </aside>

      <main className="glass grid h-full min-h-0 gap-6 rounded-[2rem] p-6 shadow-[0_22px_60px_rgba(55,31,19,0.12)] xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <section className="min-h-0 space-y-4">
          <PulseBlock className="h-4 w-28 rounded-full" />
          <PulseBlock className="h-14 w-72" />
          <PulseBlock className="h-5 w-full max-w-[34rem]" />
          <PulseBlock className="h-5 w-full max-w-[26rem]" />
          <div className="grid gap-4 md:grid-cols-2">
            <PulseBlock className="h-40 w-full" />
            <PulseBlock className="h-40 w-full" />
          </div>
        </section>

        <section className="min-h-0 space-y-4">
          <PulseBlock className="h-56 w-full" />
          <PulseBlock className="h-44 w-full" />
        </section>
      </main>
    </div>
  );
}

export function ChatRouteLoading() {
  return (
    <div className="grid h-[100dvh] max-w-[1600px] gap-4 overflow-hidden px-4 py-4 lg:grid-cols-[78px_320px_minmax(0,1fr)] lg:px-6">
      <aside className="glass flex h-full min-h-0 flex-col items-center justify-between rounded-[2rem] px-3 py-4 shadow-[0_22px_60px_rgba(55,31,19,0.12)]">
        <div className="flex flex-col items-center gap-3">
          <PulseBlock className="h-12 w-12 rounded-[1.3rem]" />
          <div className="h-px w-8 bg-[rgba(91,58,38,0.12)]" />
          <div className="flex flex-col gap-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <PulseBlock key={index} className="h-12 w-12 rounded-[1.1rem]" />
            ))}
          </div>
        </div>
        <PulseBlock className="h-12 w-12 rounded-[1.1rem]" />
      </aside>

      <aside className="glass flex h-full min-h-0 flex-col rounded-[2rem] p-4 shadow-[0_22px_60px_rgba(55,31,19,0.12)]">
        <PulseBlock className="h-4 w-24 rounded-full" />
        <PulseBlock className="mt-3 h-10 w-44" />
        <div className="app-scrollbar mt-6 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
          {Array.from({ length: 6 }).map((_, index) => (
            <PulseBlock key={index} className="h-20 w-full" />
          ))}
        </div>
      </aside>

      <main className="grid min-h-0 gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">
        <section className="glass flex min-h-0 flex-col rounded-[2rem] p-5 shadow-[0_22px_60px_rgba(55,31,19,0.12)]">
          <div className="border-b border-[rgba(91,58,38,0.08)] pb-4">
            <PulseBlock className="h-4 w-20 rounded-full" />
            <PulseBlock className="mt-3 h-12 w-64" />
            <PulseBlock className="mt-3 h-5 w-full max-w-[28rem]" />
          </div>
          <div className="min-h-0 flex-1 space-y-4 py-5">
            <PulseBlock className="h-20 w-[78%]" />
            <PulseBlock className="ml-auto h-24 w-[72%]" />
            <PulseBlock className="h-24 w-[82%]" />
          </div>
          <div className="shrink-0 rounded-[1.8rem] border border-[rgba(91,58,38,0.1)] bg-white/80 p-3">
            <PulseBlock className="h-28 w-full rounded-[1.4rem]" />
            <div className="mt-3 flex items-center justify-between gap-3 border-t border-[rgba(91,58,38,0.08)] pt-3">
              <PulseBlock className="h-4 w-full max-w-[30rem] rounded-full" />
              <PulseBlock className="h-10 w-28 rounded-full" />
            </div>
          </div>
        </section>

        <aside className="space-y-4">
          <PulseBlock className="h-36 w-full" />
          <PulseBlock className="h-52 w-full" />
          <PulseBlock className="h-44 w-full" />
        </aside>
      </main>
    </div>
  );
}

export function AuthRouteLoading() {
  return (
    <main className="mx-auto flex min-h-screen max-w-[1240px] items-center px-4 py-8 lg:px-6">
      <div className="grid w-full gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="glass rounded-[2.2rem] p-8 shadow-[0_24px_70px_rgba(55,31,19,0.12)]">
          <PulseBlock className="h-4 w-28 rounded-full" />
          <PulseBlock className="mt-5 h-20 w-full max-w-[34rem]" />
          <PulseBlock className="mt-4 h-5 w-full max-w-[30rem]" />
          <PulseBlock className="mt-2 h-5 w-full max-w-[24rem]" />
          <div className="mt-8 rounded-[1.6rem] border border-[rgba(91,58,38,0.1)] bg-white/80 p-5">
            <PulseBlock className="h-4 w-16 rounded-full" />
            <PulseBlock className="mt-3 h-12 w-full rounded-[1rem]" />
            <PulseBlock className="mt-4 h-10 w-36 rounded-full" />
            <PulseBlock className="mt-4 h-5 w-full max-w-[22rem]" />
          </div>
        </section>

        <aside className="rounded-[2.2rem] bg-[linear-gradient(155deg,#251713,#5d4135)] p-8 text-[var(--paper)] shadow-[0_26px_70px_rgba(53,29,18,0.26)]">
          <PulseBlock className="h-4 w-24 rounded-full bg-[rgba(255,239,219,0.12)]" />
          <PulseBlock className="mt-5 h-5 w-full max-w-[22rem] rounded-full bg-[rgba(255,239,219,0.12)]" />
          <PulseBlock className="mt-2 h-5 w-full max-w-[24rem] rounded-full bg-[rgba(255,239,219,0.12)]" />
          <PulseBlock className="mt-8 h-5 w-28 rounded-full bg-[rgba(255,239,219,0.12)]" />
        </aside>
      </div>
    </main>
  );
}
