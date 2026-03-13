import { END, START, StateGraph } from "@langchain/langgraph";

import {
  graphInputSchema,
  graphOutputSchema,
  graphStateSchema,
} from "@/lib/companion-foundation";
import {
  classificationNode,
  guardBlockedNode,
  guardNode,
  injectionBlockedNode,
  promptInjectionNode,
  resourceSearchNode,
  resourceSelectionNode,
  routeAfterGuard,
  routeAfterInjectionCheck,
  routeAfterTitle,
  sanitizeContextNode,
  therapistNode,
  titleNode,
  wellnessNode,
} from "@/lib/companion-nodes";

export const companionGraph = new StateGraph({
  input: graphInputSchema,
  output: graphOutputSchema,
  state: graphStateSchema,
})
  .addNode("sanitize_context_node", sanitizeContextNode)
  .addNode("detect_injection_node", promptInjectionNode)
  .addNode("guard_node", guardNode)
  .addNode("title_node", titleNode)
  .addNode("classification_node", classificationNode)
  .addNode("wellness_node", wellnessNode)
  .addNode("search_resources_node", resourceSearchNode)
  .addNode("select_resources_node", resourceSelectionNode)
  .addNode("therapist_node", therapistNode)
  .addNode("injection_block_node", injectionBlockedNode)
  .addNode("guard_block_node", guardBlockedNode)
  .addEdge(START, "sanitize_context_node")
  .addEdge("sanitize_context_node", "detect_injection_node")
  .addConditionalEdges("detect_injection_node", routeAfterInjectionCheck, [
    "guard_node",
    "injection_block_node",
  ])
  .addConditionalEdges("guard_node", routeAfterGuard, [
    "title_node",
    "classification_node",
    "guard_block_node",
  ])
  .addConditionalEdges("title_node", routeAfterTitle, [
    "classification_node",
    "guard_block_node",
  ])
  .addEdge("classification_node", "wellness_node")
  .addEdge("wellness_node", "search_resources_node")
  .addEdge("search_resources_node", "select_resources_node")
  .addEdge("select_resources_node", "therapist_node")
  .addEdge("therapist_node", END)
  .addEdge("injection_block_node", END)
  .addEdge("guard_block_node", END)
  .compile();

export async function runCompanionFlow(input: {
  userMessage: string;
  history: Array<{ role: "user" | "assistant"; content: string }>;
  wantsTitle?: boolean;
}) {
  return companionGraph.invoke(input);
}

export async function getCompanionGraphMermaid() {
  const drawableGraph = await companionGraph.getGraphAsync();
  return drawableGraph.drawMermaid();
}
