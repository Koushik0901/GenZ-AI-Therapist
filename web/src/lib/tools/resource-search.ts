import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { searchWeb, buildResourceQuery, sanitizeSearchResults, normalizeResources } from '@/lib/companion-foundation';
import { logger, logToolCall } from '@/lib/logging';
import type { Classification } from './classification';
import type { SessionTypeDetection } from './session-type';
import type { CrisisEvaluation } from './crisis-eval';
import type { ResourceItem } from '@/lib/companion-foundation';

/**
 * Conditional Resource Search Tool
 * Intelligently decides when to search, what to search for, and how to filter results
 * Skips search for venting/validation (doesn't help); deep searches for info/problem-solving
 */

export const ResourceSearchDecisionSchema = z.object({
  should_search: z.boolean().describe('Whether to search for resources'),
  search_depth: z.enum(['skip', 'minimal', 'moderate', 'deep']),
  search_query: z.string().optional(),
  reasoning: z.string(),
  skip_reason: z.string().optional(),
});

export const ResourceSearchResultSchema = z.object({
  resources: z.array(
    z.object({
      url: z.string().url(),
      description: z.string(),
      relevance_score: z.number().min(0).max(100),
    })
  ),
  decision: ResourceSearchDecisionSchema,
  confidence: z.number().min(0).max(100),
  reasoning: z.string(),
});

export type ResourceSearchDecision = z.infer<typeof ResourceSearchDecisionSchema>;
export type ResourceSearchResult = z.infer<typeof ResourceSearchResultSchema>;

const RESOURCE_SEARCH_PROMPT = `You are an expert at determining whether and how to search for mental health resources.

DECISION RULES:

SKIP SEARCH if:
- VENTING: User needs emotional release, not resources. Suggesting resources feels dismissive.
- VALIDATION: User needs affirmation and to be heard, not resources.
- CHITCHAT: Light conversation, not a place for heavy resources.
- LOW RELEVANCE: Message is off-topic or not seeking help.

MINIMAL SEARCH if:
- INFORMATION: User asked a specific question. Search for direct answers only.

MODERATE SEARCH if:
- PROBLEM-SOLVING (low urgency): User wants solutions but not in crisis. General mental health resources.

DEEP SEARCH if:
- CRISIS: Immediate crisis resources, crisis hotlines, emergency support.
- PROBLEM-SOLVING (high urgency): User wants concrete steps urgently.

Provide:
1. should_search: boolean
2. search_depth: skip | minimal | moderate | deep
3. search_query: What to search for (if searching)
4. reasoning: Why this decision
5. skip_reason: If not searching, why skip resources now

Return ONLY valid JSON (no markdown, no code blocks):
{
  "should_search": true/false,
  "search_depth": "skip|minimal|moderate|deep",
  "search_query": "query string or null",
  "reasoning": "explanation",
  "skip_reason": "optional"
}`;

/**
 * Determine whether and how to search for resources
 */
export async function decideResourceSearch(args: {
  userMessage: string;
  classification: Classification;
  sessionType: SessionTypeDetection;
  crisis: CrisisEvaluation;
}): Promise<ResourceSearchDecision> {
  // Quick heuristic-based decision first
  const quick = quickDecideResourceSearch(args);

  // If high confidence, return early
  if (quick.search_depth !== 'skip' && quick.search_query) {
    return quick;
  }

  // For ambiguous cases, use LLM
  try {
    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `Message: "${args.userMessage}"
Classification: sentiment=${args.classification.sentiment}, intent=${args.classification.intent}, confidence=${args.classification.confidence}
Session type: ${args.sessionType.primary_type} (confidence ${args.sessionType.confidence}%)
Crisis severity: ${args.crisis.severity}

${RESOURCE_SEARCH_PROMPT}`,
      temperature: 0.1,
      maxTokens: 300,
    });

    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid JSON response');
    }

    return ResourceSearchDecisionSchema.parse(JSON.parse(jsonMatch[0]));
  } catch (error) {
    logger.warn(
      {
        type: 'resource_search_decision_fallback',
        error: error instanceof Error ? error.message : String(error),
      },
      'Resource search decision failed, using fallback'
    );

    return quick;
  }
}

/**
 * Quick heuristic-based search decision
 */
function quickDecideResourceSearch(args: {
  userMessage: string;
  classification: Classification;
  sessionType: SessionTypeDetection;
  crisis: CrisisEvaluation;
}): ResourceSearchDecision {
  // Crisis always gets deep search
  if (args.crisis.severity === 'critical' || args.crisis.severity === 'high_risk') {
    return {
      should_search: true,
      search_depth: 'deep',
      search_query: buildResourceQuery(
        args.userMessage,
        args.classification.sentiment,
        args.classification.intent
      ),
      reasoning: 'Crisis detected: prioritize immediate crisis resources',
    };
  }

  // Skip for venting and validation
  if (args.sessionType.primary_type === 'venting') {
    return {
      should_search: false,
      search_depth: 'skip',
      reasoning: 'Venting session: user needs to be heard, not resources',
      skip_reason: 'Offering resources during venting can feel dismissive',
    };
  }

  if (args.sessionType.primary_type === 'validation_seeking') {
    return {
      should_search: false,
      search_depth: 'skip',
      reasoning: 'Validation-seeking session: user needs affirmation first',
      skip_reason: 'Jumping to resources bypasses the need for validation',
    };
  }

  // Problem-solving gets moderate to deep search
  if (args.sessionType.primary_type === 'problem_solving') {
    const depth = args.crisis.severity === 'at_risk' ? 'deep' : 'moderate';
    return {
      should_search: true,
      search_depth: depth,
      search_query: buildResourceQuery(
        args.userMessage,
        args.classification.sentiment,
        args.classification.intent
      ),
      reasoning: `Problem-solving session (${depth} search): user wants solutions`,
    };
  }

  // Information-seeking gets minimal to moderate search
  if (args.sessionType.primary_type === 'information_seeking') {
    return {
      should_search: true,
      search_depth: 'minimal',
      search_query: buildResourceQuery(
        args.userMessage,
        args.classification.sentiment,
        args.classification.intent
      ),
      reasoning: 'Information-seeking: provide direct answers and resources',
    };
  }

  // Chitchat: skip
  if (args.sessionType.primary_type === 'chitchat') {
    return {
      should_search: false,
      search_depth: 'skip',
      reasoning: 'Chitchat session: focus on connection, not resources',
      skip_reason: 'Heavy resources inappropriate for light conversation',
    };
  }

  // Default: moderate search
  return {
    should_search: true,
    search_depth: 'moderate',
    search_query: buildResourceQuery(
      args.userMessage,
      args.classification.sentiment,
      args.classification.intent
    ),
    reasoning: 'Default: moderate resource search',
  };
}

/**
 * Execute conditional resource search and return curated results
 */
export async function conditionalResourceSearch(args: {
  userMessage: string;
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  classification: Classification;
  sessionType: SessionTypeDetection;
  crisis: CrisisEvaluation;
}): Promise<ResourceSearchResult> {
  const startTime = Date.now();

  try {
    // Step 1: Decide whether and how to search
    const decision = await decideResourceSearch({
      userMessage: args.userMessage,
      classification: args.classification,
      sessionType: args.sessionType,
      crisis: args.crisis,
    });

    // Step 2: Skip if decided
    if (!decision.should_search || decision.search_depth === 'skip') {
      const duration = Date.now() - startTime;
      logToolCall({
        tool_name: 'resource_search',
        input: { session_type: args.sessionType.primary_type },
        output: { skipped: true, reason: decision.skip_reason },
        duration_ms: duration,
        success: true,
      });

      return {
        resources: [],
        decision,
        confidence: 92,
        reasoning: `Search skipped for ${args.sessionType.primary_type} session`,
      };
    }

    // Step 3: Execute search based on depth
    const query = decision.search_query || '';
    let resources: ResourceItem[] = [];

    if (decision.search_depth === 'minimal') {
      // Quick search for direct answer
      const results = await searchWeb(query);
      resources = normalizeResources(sanitizeSearchResults(results).slice(0, 1) as any);
    } else if (decision.search_depth === 'moderate') {
      // Standard search for general resources
      const results = await searchWeb(query);
      resources = normalizeResources(sanitizeSearchResults(results).slice(0, 2) as any);
    } else if (decision.search_depth === 'deep') {
      // Comprehensive search for crisis resources
      const results = await searchWeb(query);
      resources = normalizeResources(sanitizeSearchResults(results).slice(0, 3) as any);
    }

    // Step 4: Score resource relevance using LLM
    const scoredResources = await scoreResourceRelevance(
      resources,
      args.userMessage,
      args.sessionType.primary_type
    );

    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'resource_search',
      input: {
        session_type: args.sessionType.primary_type,
        search_depth: decision.search_depth,
      },
      output: {
        resources_found: scoredResources.length,
        avg_relevance: scoredResources.length
          ? Math.round(
              scoredResources.reduce((sum, r) => sum + r.relevance_score, 0) /
                scoredResources.length
            )
          : 0,
      },
      duration_ms: duration,
      success: true,
    });

    return {
      resources: scoredResources,
      decision,
      confidence: 85,
      reasoning: `${decision.search_depth} search executed for ${args.sessionType.primary_type} session`,
    };
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'resource_search_error',
        error: error instanceof Error ? error.message : String(error),
      },
      'Resource search failed'
    );

    logToolCall({
      tool_name: 'resource_search',
      input: { session_type: args.sessionType.primary_type },
      output: { error: true },
      duration_ms: duration,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    });

    return {
      resources: [],
      decision: {
        should_search: false,
        search_depth: 'skip',
        reasoning: 'Search error occurred',
        skip_reason: 'Resource search failed',
      },
      confidence: 0,
      reasoning: 'Resource search encountered an error',
    };
  }
}

/**
 * Score resources by relevance to user's needs
 */
async function scoreResourceRelevance(
  resources: ResourceItem[],
  userMessage: string,
  sessionType: string
): Promise<(ResourceItem & { relevance_score: number })[]> {
  if (resources.length === 0) {
    return [];
  }

  try {
    const resourceList = resources.map((r) => `- ${r.description} (${r.url})`).join('\n');

    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `User message: "${userMessage}"
Session type: ${sessionType}

Resources found:
${resourceList}

For each resource, score 0-100 relevance to the user's actual needs in this session.
Return JSON array with relevance scores:
[
  { "url": "...", "relevance_score": 0-100 },
  ...
]`,
      temperature: 0.2,
      maxTokens: 300,
    });

    const jsonMatch = response.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      // If scoring fails, return with neutral scores
      return resources.map((r) => ({ ...r, relevance_score: 50 }));
    }

    const scores = JSON.parse(jsonMatch[0]) as Array<{
      url: string;
      relevance_score: number;
    }>;

    // Map scores back to resources
    return resources
      .map((resource) => {
        const score = scores.find((s) => s.url === resource.url);
        return {
          ...resource,
          relevance_score: Math.min(100, Math.max(0, score?.relevance_score ?? 50)),
        };
      })
      .sort((a, b) => b.relevance_score - a.relevance_score);
  } catch (error) {
    // Return resources with neutral scores on error
    return resources.map((r) => ({ ...r, relevance_score: 50 }));
  }
}

/**
 * Get default empty search result
 */
export function getDefaultResourceSearch(): ResourceSearchResult {
  return {
    resources: [],
    decision: {
      should_search: false,
      search_depth: 'skip',
      reasoning: 'Default safe behavior: no search',
    },
    confidence: 100,
    reasoning: 'Default empty search result',
  };
}
