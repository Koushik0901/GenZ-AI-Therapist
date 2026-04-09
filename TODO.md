# GenZ AI Therapist: Technical Debt & Enhancement Roadmap

This document outlines all known technical debt, limitations, and planned enhancements for the GenZ AI Therapist application.

---

## 🔴 High Priority Issues

### 1. Incomplete Memory System
**Status**: Partially implemented  
**Severity**: High  
**Description**: The memory system infrastructure exists (storage, approval UI) but is non-functional.

**Current state**:
- ✅ `memory_items` table with status workflow (pending/approved/hidden)
- ✅ `getMemoryItems()` query function
- ✅ PATCH `/api/memory/[id]` endpoint for status updates
- ❌ No extraction pipeline (auto-extraction from chat not implemented)
- ❌ Not integrated into personalization (approved memories not fed to therapist)

**Why it matters**: Memory items are stored but never extracted from conversations or used to personalize responses. The UI shows "Approved memories are ready to personalize future replies once the memory layer is connected" as placeholder text.

**Implementation needed**:
1. Add extraction logic in `therapist_node` to identify user preferences/patterns
2. Create `POST /api/memory` endpoint to save extracted items as `pending`
3. Feed approved memories into therapist system prompt
4. Add UI prompt when memory candidates are identified
5. Test with real conversations to validate extraction quality

**Files involved**: 
- `web/src/lib/companion-nodes.ts` (therapist_node)
- `web/src/app/api/memory/route.ts` (add POST handler)
- `web/src/components/memory-controls.tsx` (add extraction UI)
- `web/src/lib/companion-foundation.ts` (update THERAPIST_PROMPT)

**Estimated effort**: 4-6 hours

---

### 2. Synchronous, Sequential Execution Bottleneck
**Status**: Identified  
**Severity**: High (Performance)  
**Description**: All LLM calls execute sequentially, blocking each other. This adds 15-35 seconds latency per message.

**Current flow**:
```
sanitize → detect_injection → guard → title → classification → wellness 
→ search_resources → select_resources → therapist → done
```

Each step waits for the previous to complete. Guard + Classification could run in parallel. Search + Classification could run in parallel.

**Why it matters**: 
- User waits 15-35 seconds for response
- Poor UX for fast-paced conversation
- Wastes compute on sequential waits

**Potential optimizations**:
1. **Parallelization**: Run guard + title in parallel after injection check
2. **Parallelization**: Run classification + wellness in parallel
3. **Parallelization**: Run search_resources while classification runs
4. **Streaming**: Stream therapist response token-by-token instead of waiting for full text
5. **Lazy loading**: Defer resource search/selection until response is already visible

**Estimated improvement**:
- Current: 15-35s
- With parallelization: 5-15s
- With streaming: 2-3s before first tokens appear

**Files involved**:
- `web/src/lib/companion-graph.ts` (modify edges, add parallel routes)
- `web/src/app/api/chat/send/route.ts` (handle streaming responses)
- `web/src/components/assistant-thread.tsx` (stream UI updates)

**Estimated effort**: 6-8 hours

---

### 3. No Streaming Responses
**Status**: Not implemented  
**Severity**: High (UX)  
**Description**: Full assistant response is generated before sending to client. User waits 5-7 seconds for therapist node alone.

**Current behavior**:
```
POST /api/chat/send
  → Run full graph (15-35s)
  → Get complete response text
  → Send JSON
  → Client renders all at once
```

**Desired behavior**:
```
POST /api/chat/send
  → Run graph until therapist node
  → Start streaming therapist response
  → Client renders tokens as they arrive
  → User sees response forming in real-time
```

**Why it matters**:
- Perceived latency: 5-7s with streaming vs. 15-35s without
- More engaging UX (feels responsive)
- Better for long responses

**Implementation approach**:
1. Modify therapist_node to stream to client
2. Use Server-Sent Events (SSE) or Streaming Response in Next.js
3. Update `assistant-thread.tsx` to append tokens as they arrive
4. Fallback: Continue with current behavior if streaming fails

**Files involved**:
- `web/src/app/api/chat/send/route.ts` (implement streaming)
- `web/src/lib/companion-nodes.ts` (therapist_node streaming)
- `web/src/components/assistant-thread.tsx` (stream handler)

**Estimated effort**: 4-5 hours

---

## 🟡 Medium Priority Issues

### 4. Client-Side Analytics Computation
**Status**: Identified  
**Severity**: Medium (Performance)  
**Description**: All insights are calculated in JavaScript on every request. No caching, no SQL views.

**Current approach**:
```
GET /api/insights
  → Fetch raw check-ins (14 max)
  → Fetch raw messages (24 max)
  → Fetch session counts
  → Fetch journal counts
  → Compute trends in JS
  → Compute streaks in JS
  → Build insight cards in JS
  → Return JSON
```

**Problems**:
1. Repeated computation on every request (no caching)
2. Trends recalculated if viewed multiple times
3. Complex JavaScript logic instead of SQL (harder to test, debug)
4. Could become slow as data grows

**Why it matters**:
- Insights page may slow down as user has more data
- Extra compute on Vercel (cold starts)
- Hard to version/A/B test analytics logic

**Better approaches**:
1. **SQL Views**: Create materialized or computed views for common queries
2. **Caching**: Cache insights for 1 hour (trends don't need real-time updates)
3. **Scheduled jobs**: Pre-compute insights at midnight
4. **Move to Supabase functions**: Compute in PostgreSQL, return JSON

**Recommended approach**:
1. Extract trend computation to Supabase functions (PL/pgSQL)
2. Cache results in Redis or application memory (1 hour TTL)
3. Invalidate cache on new check-in or message
4. Keep JavaScript as fallback if function fails

**Files involved**:
- `web/src/lib/insights.ts` (move logic to Supabase)
- `supabase/migrations/0001_init.sql` (add functions)
- `web/src/app/api/insights/route.ts` (call functions instead)

**Estimated effort**: 3-4 hours

---

### 5. Limited Context Window
**Status**: Identified  
**Severity**: Medium (Feature)  
**Description**: Only last 8 messages (4 exchanges) are sent to LLM. Longer conversations lose early context.

**Current:**
```typescript
// In companion-nodes.ts: sanitizeContextNode
history.slice(-8)  // Only last 8 messages
```

**Problems**:
1. User context from 20+ messages ago is forgotten
2. LLM can't see long-term patterns (e.g., "I've tried this 3 times already")
3. Relationships/callbacks don't carry across days

**Why it's limited**:
- Token budget: 8 messages ≈ 2-3k tokens context
- Sanitization: Each message capped at 700 chars
- OpenRouter cost: More context = higher cost

**Solutions**:
1. **Hierarchical summarization**: Summarize older messages, keep recent verbatim
2. **Smart pruning**: Keep only emotionally significant messages
3. **Session summary**: Generate summary at start of new day
4. **External store**: Retrieve relevant memories instead of full history

**Recommended**: Implement smart pruning with emotional significance scoring
```
If message.sentiment in ["Crisis", "Positive"] or 
   message.wellness.mood < 20 or message.wellness.stress > 80:
  Keep message
Else:
  Candidate for pruning
```

**Files involved**:
- `web/src/lib/companion-foundation.ts` (add scoring)
- `web/src/lib/companion-nodes.ts` (modify sanitizeContextNode)

**Estimated effort**: 2-3 hours

---

### 6. Hardcoded System Prompts
**Status**: Identified  
**Severity**: Medium (Maintainability)  
**Description**: All system prompts (GUARD_PROMPT, THERAPIST_PROMPT, etc.) are embedded in `companion-foundation.ts`. This makes A/B testing and prompt management difficult.

**Current state**:
```typescript
// In companion-foundation.ts
export const GUARD_PROMPT = "You are a...";  // 200+ lines
export const THERAPIST_PROMPT = "...";       // 400+ lines
// ... 6 more prompts embedded
```

**Problems**:
1. Hard to A/B test different prompt versions
2. Changes require code redeployment
3. Version control mixes code + content
4. No easy rollback if prompt change breaks things
5. Hard to track which prompt version is live

**Better approach**:
1. Store prompts in external system (e.g., Supabase table, environment variables, JSON files)
2. Load at runtime (with caching)
3. Allow hot updates without redeployment
4. Version prompts separately from code

**Implementation**:
```sql
-- Add prompts table
CREATE TABLE prompts (
  id uuid PRIMARY KEY,
  name text UNIQUE,
  content text,
  version integer,
  active boolean,
  created_at timestamp,
  updated_at timestamp
);
```

```typescript
// Load at startup
const prompts = await loadPromptsFromSupabase();
const GUARD_PROMPT = prompts.get("guard_prompt_v2");
```

**Benefits**:
- Hot updates (no redeploy)
- Easy A/B testing
- Version history
- Quick rollback

**Files involved**:
- `supabase/migrations/0001_init.sql` (add table)
- `web/src/lib/companion-foundation.ts` (move prompts, add loader)
- `web/src/app/api/admin/prompts/` (admin endpoints to manage)

**Estimated effort**: 2-3 hours

---

### 7. No Streaming in Wellness/Resource Selection
**Status**: Identified  
**Severity**: Medium (UX)  
**Description**: Resources and wellness inference block the therapist node. If resource search is slow, entire response is delayed.

**Current flow**:
```
wellness_node (wait) → search_resources (wait, slow!) → 
select_resources (wait) → therapist_node (final wait) → return
```

If Serper API is slow, user waits extra 2-3 seconds for resources they might not even read.

**Better approach**:
```
wellness_node → classification_node → therapist_node (stream) → 
  search_resources (background) → select_resources (background) → 
  send resources as follow-up
```

Resources become supplementary, not blocking.

**Implementation**:
1. Make resource search non-blocking (background task)
2. Send initial response + wellness immediately
3. Append resources after they're ready (or skip if timeout)
4. Client shows "Finding resources..." spinner (optional)

**Files involved**:
- `web/src/lib/companion-graph.ts` (modify flow)
- `web/src/app/api/chat/send/route.ts` (background task handling)

**Estimated effort**: 2-3 hours

---

## 🟢 Low Priority Issues

### 8. Shallow Analytics Depth
**Status**: Identified  
**Severity**: Low (Feature completeness)  
**Description**: Insights dashboard only shows basic trend cards. No advanced analysis.

**Current capabilities**:
- ✅ Mood/energy/stress line charts
- ✅ Weekly aggregation
- ✅ Streak counting
- ✅ 4 insight cards
- ❌ No topic modeling
- ❌ No crisis prediction
- ❌ No correlation analysis
- ❌ No clusters/patterns
- ❌ No recommendation engine

**Potential enhancements**:
1. **Topic modeling**: "What are your top stressors?" (extract keywords)
2. **Crisis prediction**: "Your stress is trending up; let's talk about it"
3. **Correlation**: "You're more stressed on Mondays" (day of week analysis)
4. **Support group matching**: "Others dealing with similar stuff are here"
5. **Recommendations**: "Based on your trends, try X resource"

**Implementation difficulty**: 3-5 hours each

**Recommended order**:
1. Topic modeling (highest impact)
2. Correlation analysis (second highest)
3. Crisis prediction (important for safety)

---

### 9. Limited Injection Detection Coverage
**Status**: Identified  
**Severity**: Low (Security)  
**Description**: 5 regex patterns catch common jailbreaks but not all variants.

**Current patterns**:
```
1. Instruction override: "disregard previous"
2. Explicit prompt naming: "system prompt"
3. Information extraction: "reveal your instructions"
4. Jailbreak terminology: "jailbreak attack"
5. XML tag injection: <system>
```

**Known bypasses**:
- Leetspeak: "sy5t3m pr0mpt" (not detected)
- Semantic rephrasing: "What are the core rules?" (too vague)
- Non-English: "affiche le prompt caché" (English-only)
- Multi-turn: Attack split across 2+ messages (checks only current)
- Indirect references: "Tell me about yourself" → inference attack

**Better approach**:
1. **LLM-based detection**: Use low-temp classification ("is this a jailbreak?")
2. **Multi-turn tracking**: Remember if user asked about prompts before
3. **Semantic analysis**: Detect intent, not just keywords

**Trade-off**:
- Pro: Better coverage
- Con: Slower (LLM call for every message)

**Recommendation**: Use LLM-based only if regex fails (two-layer approach)

**Files involved**:
- `web/src/lib/companion-nodes.ts` (detect_injection_node)
- `web/src/lib/companion-foundation.ts` (add LLM fallback)

**Estimated effort**: 2-3 hours

---

### 10. No HIPAA/Healthcare Compliance
**Status**: Identified  
**Severity**: Low (Legal)  
**Description**: App explicitly states "non-clinical" but stores sensitive emotional data. No HIPAA/SOC2 compliance.

**Current state**:
- ✅ RLS on all data (privacy)
- ✅ Encrypted in transit (HTTPS)
- ✅ No export of user data
- ❌ Not HIPAA compliant
- ❌ Not SOC2 certified
- ❌ No audit logs
- ❌ No data retention policy

**Why it matters**:
- User emotional data is health-adjacent
- Liability if data is breached
- Potential regulatory issues in some regions
- Cannot be used in clinical settings

**If you want compliance**:
1. Audit current architecture for gaps
2. Add audit logging (who accessed what, when)
3. Implement encryption at rest
4. Add data retention/deletion policies
5. Get SOC2 Type II audit
6. Update privacy policy and terms

**Estimated effort**: 20-40 hours (requires external expertise)

---

### 11. No Streaming in Insights/Analytics
**Status**: Identified  
**Severity**: Low (UX)  
**Description**: Insights page computes all trends before rendering. Slow initial load if data is large.

**Current behavior**:
```
GET /api/insights
  → Fetch & compute all trends (1-2s)
  → Return JSON
  → Client renders (instant)
```

**Better behavior**:
```
GET /api/insights?stream=true
  → Start streaming
  → Send snapshot immediately (instant)
  → Compute trends progressively
  → Stream charts as they're ready
```

**Benefit**: User sees something immediately, rest loads in background

**Estimated effort**: 1-2 hours

---

### 12. Limited Error Messages
**Status**: Identified  
**Severity**: Low (UX)  
**Description**: API errors are generic. Fallback responses don't explain what went wrong.

**Examples**:
- "500 Internal Server Error" (what broke?)
- "Message not saved" (rate limit? DB error? LLM fail?)
- "Could not retrieve insights" (why?)

**Better approach**:
```typescript
{
  ok: false,
  error: "rate_limit_exceeded",
  message: "Too many messages (8 per 60s). Try again in 45 seconds.",
  retryAfter: 45,
}
```

**Implementation**:
1. Define error code enum
2. Return specific error codes from API
3. Client displays user-friendly message
4. Log error details server-side

**Estimated effort**: 1-2 hours

---

### 13. No Pagination for Large Message Histories
**Status**: Identified  
**Severity**: Low (Performance)  
**Description**: Session messages loaded all at once. Could be slow with 100+ messages.

**Current behavior**:
```typescript
// Get all messages for session
supabase.from("messages")
  .select("*")
  .eq("session_id", sessionId)
  .order("created_at")
```

**Better approach**:
```typescript
// Paginate: first 20, then load more on scroll
supabase.from("messages")
  .select("*")
  .eq("session_id", sessionId)
  .order("created_at")
  .limit(20)
  .offset(offset)
```

**Implementation**: 
1. Add pagination params to API
2. Modify chat component to load-on-scroll
3. Append messages to DOM as user scrolls up

**Estimated effort**: 2-3 hours

---

## 📋 Maintenance & Code Quality Issues

### 14. No Automated Testing
**Status**: None detected  
**Severity**: Medium (Quality)  
**Description**: No unit tests, integration tests, or E2E tests visible.

**Recommended test coverage**:
- **Unit tests**: Node functions, helper utilities (40-50% of codebase)
- **Integration tests**: API endpoints, database queries (20-30%)
- **E2E tests**: User flows (chat, journal, insights) (10-15%)

**Key test areas**:
1. Fallback logic (does it work when LLM fails?)
2. Rate limiting (is it enforced?)
3. RLS policies (can user see others' data?)
4. Node routing (do edges route correctly?)
5. Wellness calculation (is scoring correct?)

**Setup**:
```bash
# Add testing framework
npm install --save-dev vitest @testing-library/react

# Create test files
web/src/lib/__tests__/companion-nodes.test.ts
web/src/lib/__tests__/insights.test.ts
web/src/app/api/__tests__/chat.send.test.ts
```

**Estimated effort**: 20-30 hours to achieve 60% coverage

---

### 15. Incomplete Error Logging
**Status**: Identified  
**Severity**: Medium (Observability)  
**Description**: No structured logging. Hard to debug production issues.

**Current state**:
```typescript
try {
  // ...
} catch (e) {
  console.error(e);  // Logs to stdout, no structure
  return NextResponse.json({error: "..."}, {status: 500});
}
```

**Better approach**:
```typescript
// Add structured logging
import { createLogger } from "@/lib/logger";
const logger = createLogger("chat-api");

try {
  // ...
} catch (e) {
  logger.error("Message send failed", {
    userId,
    sessionId,
    errorCode: e.code,
    errorMessage: e.message,
    stack: e.stack,
  });
}
```

**Options**:
1. **Vercel logs**: Use built-in logging (free, integrated)
2. **Sentry**: Error tracking + alerting (20/mo)
3. **LogRocket**: User session replay (99/mo)
4. **Datadog**: Full observability (15/mo)

**Recommended**: Start with Vercel logs + Sentry for errors

**Estimated effort**: 3-4 hours

---

### 16. No Rate Limit Testing
**Status**: Identified  
**Severity**: Low (Quality)  
**Description**: Rate limiter logic exists but is not tested. Could have gaps.

**Current implementation**:
```typescript
// In lib/rate-limit.ts
const bucket = buckets.get(key);
if (bucket.remaining < 1) {
  return { allowed: false };
}
bucket.remaining--;
```

**Potential issues**:
- Race conditions with concurrent requests?
- In-memory storage lost on process restart
- No cleanup of old buckets (memory leak?)

**Better approach**:
1. Move rate limiting to Supabase (persistent, atomic)
2. Add Redis if needed for speed
3. Add comprehensive tests

**Estimated effort**: 4-5 hours

---

## 🔧 Developer Experience Issues

### 17. Limited Documentation
**Status**: Identified  
**Severity**: Low (DX)  
**Description**: Code comments are sparse. Architecture not documented.

**Missing documentation**:
- Architecture overview (ARCHITECTURE.md)
- API documentation (OpenAPI spec)
- Database schema explanation
- Contribution guidelines
- Deployment guide
- Local development setup

**Recommended additions**:
```
/docs/
  ├── ARCHITECTURE.md         ← System design
  ├── API.md                  ← Endpoint reference
  ├── DATABASE.md             ← Schema explanation
  ├── DEPLOYMENT.md           ← How to deploy
  ├── DEVELOPMENT.md          ← Local setup
  └── PROMPTS.md              ← Prompt engineering guide
```

**Estimated effort**: 4-6 hours

---

### 18. Environment Variable Documentation
**Status**: Incomplete  
**Severity**: Low (DX)  
**Description**: `.env` variables not documented. What's required vs. optional?

**Current state**:
```bash
NEXT_PUBLIC_SUPABASE_URL         # Required but not labeled
NEXT_PUBLIC_SUPABASE_ANON_KEY    # Required but not labeled
OPENROUTER_API_KEY               # Required but not labeled
SERPER_API_KEY                   # Optional but not labeled
```

**Better approach**:
```bash
# Create .env.example
# Required (app won't work without these)
NEXT_PUBLIC_SUPABASE_URL=https://...
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...

# Required (for LLM)
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=minimax/minimax-m2.5

# Optional (resource discovery, leave blank to skip)
SERPER_API_KEY=sk-...

# Optional (URLs, fallback to defaults)
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

**Estimated effort**: 30 minutes

---

## 📊 Optional Enhancements (Nice-to-Have)

### 19. Dark Mode Support
**Status**: Not implemented  
**Severity**: Very Low (UX)  
**Description**: No dark mode theme. Only light mode available.

**Implementation**: Add Tailwind dark mode via `tailwind.config.ts`

**Estimated effort**: 1-2 hours

---

### 20. Multi-Language Support (i18n)
**Status**: Not implemented  
**Severity**: Very Low (Localization)  
**Description**: All text is English. No i18n framework.

**Implementation**: Add `next-i18next` or similar

**Estimated effort**: 4-6 hours

---

### 21. Export Data / Backup
**Status**: Not implemented  
**Severity**: Low (Feature)  
**Description**: Users cannot export their data (conversations, journals, insights).

**Implementation**:
1. Add "Export" button in settings
2. Generate JSON/CSV of all user data
3. Download zip file

**Estimated effort**: 2-3 hours

---

### 22. Time Zone Handling
**Status**: Not implemented  
**Severity**: Low (Feature)  
**Description**: All timestamps are UTC. No user time zone awareness.

**Current behavior**:
- Check-in recorded at "2025-04-05T14:32:00Z"
- Displayed as UTC (confusing if user is in PST)

**Better**:
- Store timezone in profile
- Display times in user's timezone
- Compute streaks in user's timezone (not UTC)

**Implementation**: Add timezone picker in settings, use `date-fns-tz`

**Estimated effort**: 2-3 hours

---

### 23. Offline Support
**Status**: Not implemented  
**Severity**: Very Low (Feature)  
**Description**: App requires internet connection. No offline mode.

**Implementation**:
1. Add Service Worker for offline caching
2. Queue messages when offline
3. Sync when reconnected

**Estimated effort**: 3-4 hours

---

### 24. Progressive Web App (PWA)
**Status**: Not implemented  
**Severity**: Very Low (Feature)  
**Description**: Not installable as PWA. Only web-based.

**Implementation**: Add manifest.json + service worker

**Estimated effort**: 1-2 hours

---

## 🎯 Recommended Implementation Order

### Phase 1: Critical Fixes (1-2 weeks)
1. **Memory system extraction** (#1) — High impact, needed for marketing
2. **Synchronous execution** (#2) — UX improvement, high impact
3. **Streaming responses** (#3) — UX improvement, medium effort

### Phase 2: Performance & Quality (2-3 weeks)
4. **Client-side analytics** (#4) — Future-proofing
5. **Automated testing** (#14) — Quality assurance
6. **Error logging** (#15) — Observability

### Phase 3: Polish & Features (3-4 weeks)
7. **Context window** (#5) — Feature completeness
8. **Hardcoded prompts** (#6) — Maintainability
9. **Analytics depth** (#8) — Feature completeness
10. **Documentation** (#17) — DX improvement

### Phase 4: Nice-to-Have (Ongoing)
- Dark mode (#19)
- Data export (#21)
- i18n (#20)
- PWA (#24)

---

## Summary Statistics

| Category | Count | Est. Effort |
|----------|-------|-------------|
| Critical (High Priority) | 3 | 15-19 hours |
| Important (Medium Priority) | 7 | 14-20 hours |
| Nice-to-Have (Low Priority) | 7 | 10-16 hours |
| Very Low (Optional) | 4 | 5-10 hours |
| **Total** | **21** | **44-65 hours** |

**Quick Wins** (1-2 hours each):
- Documentation updates (#17-18)
- Environment variable documentation
- Error message improvements (#12)
- Shallow analytics start (#8)

---

## How to Use This Document

1. **Planning sprints**: Copy high-priority items into your sprint
2. **Code reviews**: Reference specific sections when reviewing related code
3. **Onboarding**: Share with new team members to understand architectural gaps
4. **Prioritization**: Reorder based on business needs and user feedback
5. **Tracking**: Convert items to GitHub Issues with links to this document

---

**Last Updated**: April 8, 2025  
**Status**: Comprehensive tech debt audit complete  
**Next Review**: After Phase 1 completion
