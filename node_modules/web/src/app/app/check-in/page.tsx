import { CheckInPanel } from "@/components/check-in-panel";
import { getLatestCheckIn } from "@/lib/wellness";
import { getViewer } from "@/lib/viewer";

export default async function CheckInPage() {
  const { user } = await getViewer();
  const userId = user?.id;

  const checkIn = await getLatestCheckIn(userId);

  return (
    <CheckInPanel
      initialMood={checkIn.mood}
      initialEnergy={checkIn.energy}
      initialStress={checkIn.stress}
      initialNote={checkIn.note}
      demoMode={!userId}
    />
  );
}
