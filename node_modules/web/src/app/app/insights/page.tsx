import { InsightsDashboard } from "@/components/insights-dashboard";
import { getInsights } from "@/lib/insights";
import { getViewer } from "@/lib/viewer";

export default async function InsightsPage() {
  const { user } = await getViewer();
  const userId = user?.id;

  const insights = await getInsights(userId);
  return <InsightsDashboard insights={insights} />;
}
