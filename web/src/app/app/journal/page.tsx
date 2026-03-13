import { JournalStudio } from "@/components/journal-studio";
import { getJournalEntries } from "@/lib/wellness";
import { getViewer } from "@/lib/viewer";

export default async function JournalPage() {
  const { user } = await getViewer();
  const userId = user?.id;

  const entries = await getJournalEntries(userId);

  return <JournalStudio initialEntries={entries} demoMode={!userId} />;
}
