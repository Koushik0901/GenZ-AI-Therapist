# Migration Fix - Schema Conflict Resolved

## Problem Found

When running `0003_feedback_system.sql`, you got this error:

```
ERROR: 42703: column "sentiment" does not exist
```

## Root Cause

The migration files had a **table duplication issue**:

- **0002_agentic_logging.sql** creates a `feedback` table with:
  - `helpful` (boolean) column
  - References to `auth.users` and `chat_sessions`
  - Proper RLS policies

- **0003_feedback_system.sql** was trying to create a DIFFERENT `feedback` table with:
  - `sentiment` (text) column  
  - Different schema
  - This conflicted with the table from 0002

## Solution Applied

✅ **Fixed 0003_feedback_system.sql**

Removed the conflicting feedback table from migration 0003. The table now only creates:

1. **feedback_summary** - Aggregated metrics (NEW)
2. **session_metadata** - Multi-turn conversation data
3. **strategy_performance** - Strategy effectiveness tracking
4. **ab_test_variants** - A/B test variants
5. **ab_test_results** - A/B test results
6. **system_alerts** - System alerts and notifications

The basic `feedback` table is properly handled by **0002_agentic_logging.sql**.

## How to Deploy Now

### If you haven't run the migrations yet:

Run them in order:
```
1. 0001_init.sql
2. 0002_agentic_logging.sql  
3. 0003_feedback_system.sql  ← Fixed!
```

### If you already ran 0001 and 0002:

Simply run 0003 again - it should succeed now:
```sql
-- In Supabase SQL Editor, run: 0003_feedback_system.sql
```

### If 0003 already failed:

1. Delete any tables created by the failed 0003 run
2. Run 0003 again with the fixed version

## What Each Migration Does

| Migration | Purpose | Tables Created |
|-----------|---------|-----------------|
| **0001_init.sql** | Core schema | profiles, sessions, messages, journal, check-ins, memory |
| **0002_agentic_logging.sql** | Logging & feedback | tool_calls, orchestrator_decisions, response_evaluations, crisis_evaluations, feedback |
| **0003_feedback_system.sql** | Analytics | feedback_summary, session_metadata, strategy_performance, ab_test_variants, ab_test_results, system_alerts |

## Verification

After running all 3 migrations, you should have **13 tables total**:

**Core (0001):**
- profiles
- chat_sessions
- chat_messages
- journal_entries
- daily_checkins
- user_memory

**Logging (0002):**
- tool_calls
- orchestrator_decisions
- response_evaluations
- crisis_evaluations
- feedback

**Analytics (0003):**
- feedback_summary
- session_metadata
- strategy_performance
- ab_test_variants
- ab_test_results
- system_alerts

**Total: 13 tables** ✅

## Next Steps

1. Run migrations in Supabase (use fixed 0003_feedback_system.sql)
2. Verify all 13 tables appear in Supabase Table Editor
3. Continue with Vercel deployment from DEPLOYMENT-CHECKLIST.md

---

**Status**: ✅ Fixed  
**Migration File**: 0003_feedback_system.sql  
**Date Fixed**: April 9, 2026
