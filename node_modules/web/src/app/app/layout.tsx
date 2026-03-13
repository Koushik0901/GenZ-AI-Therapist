import { headers } from "next/headers";

import { signOutAction } from "@/app/auth/actions";
import { AppShell } from "@/components/app-shell";
import { getSessionSummaries } from "@/lib/chat";
import { getViewer } from "@/lib/viewer";

export default function ProductLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <ProductLayoutInner>{children}</ProductLayoutInner>;
}

async function ProductLayoutInner({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const requestHeaders = await headers();
  const pathname = requestHeaders.get("x-app-pathname") ?? "";
  const shouldLoadHistory = pathname.startsWith("/app/chat");

  const { user } = await getViewer();

  if (!user) {
    return (
      <AppShell
        userLabel="Demo user"
        recentSessions={
          shouldLoadHistory ? await getSessionSummaries(undefined) : []
        }
        signOutAction={undefined}
      >
        {children}
      </AppShell>
    );
  }
  const recentSessions = shouldLoadHistory
    ? await getSessionSummaries(user.id)
    : [];

  return (
    <AppShell
      userLabel={user.email ?? "Signed-in user"}
      recentSessions={recentSessions}
      signOutAction={signOutAction}
    >
      {children}
    </AppShell>
  );
}
