import type { SessionSummary } from "@/lib/chat";
import { AppShellFrame } from "@/components/app-shell-frame";

type AppShellProps = {
  children: React.ReactNode;
  userLabel: string;
  recentSessions?: SessionSummary[];
  signOutAction?: () => Promise<void>;
};

export function AppShell({
  children,
  userLabel,
  recentSessions = [],
  signOutAction,
}: AppShellProps) {
  return (
    <AppShellFrame
      userLabel={userLabel}
      recentSessions={recentSessions}
      signOutAction={signOutAction}
    >
      {children}
    </AppShellFrame>
  );
}
