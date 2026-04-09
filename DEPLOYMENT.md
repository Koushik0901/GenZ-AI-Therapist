# GenZ AI Therapist – Production Deployment Guide

This guide covers **complete setup, deployment, and operational monitoring** for GenZ AI Therapist in production environments.

---

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Architecture Overview](#architecture-overview)
3. [Environment Setup](#environment-setup)
4. [Database Migrations](#database-migrations)
5. [Deployment Steps](#deployment-steps)
6. [Admin Authentication](#admin-authentication)
7. [Monitoring & Observability](#monitoring--observability)
8. [Performance & Scaling](#performance--scaling)
9. [Troubleshooting](#troubleshooting)
10. [Disaster Recovery](#disaster-recovery)
11. [Security Best Practices](#security-best-practices)
12. [Testing in Production](#testing-in-production)

---

## Pre-Deployment Checklist

Before deploying to production, verify:

- [ ] All environment variables are configured
- [ ] Supabase project is created with appropriate plan
- [ ] OpenRouter and Serper API keys are active
- [ ] All database migrations have been applied
- [ ] Tests pass: `npm run test` (in web directory)
- [ ] Build succeeds: `npm run build`
- [ ] No console errors on production build
- [ ] Admin authentication token is generated and stored securely
- [ ] Vercel environment variables are set up
- [ ] Domain / DNS is configured (if custom domain)
- [ ] Rate limiting is appropriately configured
- [ ] Backup strategy is in place

---

## Architecture Overview

### Production Stack

```
┌─────────────────────────────────────────────────────────────┐
│                        End Users                             │
└────────────────┬────────────────────────────────────────────┘
                 │
         ┌───────▼────────┐
         │    Vercel      │  ◄── Next.js 16 Frontend + API
         │   (Frontend)   │
         └────────┬───────┘
                  │
         ┌────────▼────────────────────────────────────────┐
         │          API Layer (Next.js Routes)             │
         │  • /api/chat/*          (chat operations)       │
         │  • /api/metrics         (admin - protected)     │
         │  • /api/alerts/*        (admin - protected)     │
         │  • /api/sessions/*      (user analytics)        │
         │  • /api/strategies/*    (strategy recs)         │
         └────────┬─────────────────────────────────────────┘
                  │
         ┌────────▼──────────────┐
         │     Supabase          │
         │  ┌─────────────────┐  │
         │  │ PostgreSQL DB   │  │  ◄── Auth, Profiles, Sessions
         │  └─────────────────┘  │
         │  ┌─────────────────┐  │
         │  │  Auth (Magic)   │  │  ◄── User authentication
         │  └─────────────────┘  │
         │  ┌─────────────────┐  │
         │  │  Row Security   │  │  ◄── Per-user data isolation
         │  └─────────────────┘  │
         └────────┬───────────────┘
                  │
    ┌─────────────┴──────────────┬───────────────┐
    │                            │               │
┌───▼───────┐           ┌────────▼────┐   ┌─────▼─────┐
│ OpenRouter│           │  Serper     │   │ Supabase  │
│(Kimi 2.5)│           │  (Resources)│   │(Analytics)│
└───────────┘           └─────────────┘   └───────────┘
```

### Graceful Degradation

All systems work with **and without Supabase**:

- ✅ Chat still works (in-memory session state)
- ✅ Monitoring still logs locally (falls back to console)
- ✅ Alerts still trigger (console warnings)
- ✅ User preferences still learn (in-memory tracking)
- ✅ Resources still provide recommendations (without DB persistence)

---

## Environment Setup

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Wait for PostgreSQL database to initialize (~2 minutes)
4. Copy project credentials:
   - **Project URL**: `Settings → API → URL`
   - **Anon Key**: `Settings → API → Project API Keys → anon`

### 2. Configure Auth URLs in Supabase

In the Supabase dashboard:

1. Go to **Authentication → URL Configuration**
2. Set **Site URL**:
   - **Local development**: `http://localhost:3000`
   - **Production**: `https://your-domain.com`
3. Set **Redirect URLs**:
   - `http://localhost:3000/auth/callback` (local)
   - `https://your-domain.com/auth/callback` (production)

### 3. Generate Admin Authentication Token

Admin endpoints require a secure token. Generate one:

```bash
# Generate a random 64-character token
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Store this securely in your password manager. You'll add it to Vercel env vars as `ADMIN_AUTH_TOKEN`.

### 4. Create `.env` File

Create `web/.env.local` or `/.env`:

```dotenv
# ─────────────────────────────────────────
# REQUIRED: Application URLs
# ─────────────────────────────────────────
NEXT_PUBLIC_APP_URL=https://your-domain.com
APP_URL=https://your-domain.com

# ─────────────────────────────────────────
# REQUIRED: Supabase Configuration
# ─────────────────────────────────────────
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here

# ─────────────────────────────────────────
# REQUIRED: AI Model Configuration
# ─────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=minimax/minimax-m2.5
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ─────────────────────────────────────────
# OPTIONAL: Resource Search
# ─────────────────────────────────────────
SERPER_API_KEY=your_serper_key_here

# ─────────────────────────────────────────
# REQUIRED: Admin Authentication
# ─────────────────────────────────────────
ADMIN_AUTH_TOKEN=your_64_char_random_token_here
```

### 5. Verify Environment Variables

Test that your setup works:

```bash
cd web
npm run build
```

If build succeeds, your environment is correctly configured.

---

## Database Migrations

### Run Initial Schema Migration

The app requires three Supabase migrations for full functionality:

#### Migration 1: Core Tables (users, sessions, messages)

Run in Supabase SQL Editor:

```sql
-- See file: supabase/migrations/0001_init.sql
-- This creates:
-- - profiles (extended user data)
-- - chat_sessions (conversation containers)
-- - messages (individual messages)
-- - journal_entries (user journal)
-- - daily_checkins (vibe check data)
-- - memory_items (app memory)
```

#### Migration 2: Feedback & Analytics System

```sql
-- See file: supabase/migrations/0002_feedback_system.sql
-- This creates:
-- - response_feedback (user ratings on responses)
-- - session_quality_metrics (overall quality tracking)
-- - user_satisfaction_scores (rolling satisfaction tracking)
-- - api_error_logs (error tracking)
-- - crisis_escalations (crisis pattern tracking)
-- - regeneration_events (when responses are regenerated)
-- - system_alerts (admin alerts)
```

#### Migration 3: Analytics & Monitoring Tables

```sql
-- See file: supabase/migrations/0003_feedback_system.sql
-- This creates analytics tables for:
-- - Strategy performance tracking
-- - User preference learning
-- - A/B testing support
-- - Advanced monitoring
```

### Apply Migrations to Supabase

1. Open Supabase dashboard
2. Go to **SQL Editor**
3. Create a new query
4. Copy the contents of each migration file from `supabase/migrations/`
5. Run each migration in order (0001 → 0002 → 0003)
6. Verify tables appear in **Table Editor**

---

## Deployment Steps

### Option A: Deploy to Vercel (Recommended)

**Benefits:**
- Zero-config deployment
- Free HTTPS
- Automatic deployments on git push
- Environment variable UI
- Log streaming

#### Step 1: Connect GitHub Repository

1. Go to [vercel.com](https://vercel.com)
2. Click **Add New → Project**
3. Select your GitHub repository
4. Select root directory: `./web`

#### Step 2: Configure Environment Variables

In Vercel project dashboard:

1. Go to **Settings → Environment Variables**
2. Add all variables from your local `.env`:

```
NEXT_PUBLIC_APP_URL
NEXT_PUBLIC_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY
OPENROUTER_API_KEY
OPENROUTER_MODEL
OPENROUTER_BASE_URL
SERPER_API_KEY (optional)
ADMIN_AUTH_TOKEN
```

#### Step 3: Deploy

```bash
# Push to main branch triggers auto-deployment
git push origin main

# Or manually deploy from Vercel dashboard:
# Click "Deploy" button
```

Vercel provides a public URL immediately.

#### Step 4: Configure Custom Domain (Optional)

In Vercel project settings:

1. Go to **Settings → Domains**
2. Enter your domain (e.g., `therapist.yoursite.com`)
3. Follow DNS verification steps
4. Update Supabase **Site URL** to match your domain

### Option B: Self-Hosted Deployment

For AWS, DigitalOcean, Railway, etc.:

#### Step 1: Build the Application

```bash
cd web
npm install
npm run build
```

#### Step 2: Create Start Script

```bash
cd web
npm run start
```

The app listens on `PORT` env var (default: 3000).

#### Step 3: Configure Reverse Proxy (Nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name therapist.yoursite.com;

    # SSL certificates (use Let's Encrypt or your provider)
    ssl_certificate /etc/letsencrypt/live/therapist.yoursite.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/therapist.yoursite.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Proxy to Node.js app
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Step 4: Set Environment Variables on Server

Create `/home/app/.env`:

```bash
export NEXT_PUBLIC_APP_URL=https://therapist.yoursite.com
export NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
# ... all other env vars
```

#### Step 5: Use Process Manager (PM2)

```bash
npm install -g pm2

cd web
pm2 start "npm start" --name "therapist" --env /home/app/.env

# Auto-restart on reboot
pm2 startup
pm2 save
```

---

## Admin Authentication

The app uses token-based admin authentication to protect sensitive endpoints.

### Protected Endpoints

```
GET /api/metrics           (system performance metrics)
GET /api/alerts            (active system alerts)
POST /api/alerts/acknowledge (acknowledge an alert)
```

### How It Works

Requests to protected endpoints must include an `Authorization` header:

```http
GET /api/metrics HTTP/1.1
Authorization: Bearer your_admin_token_here
```

OR use custom header:

```http
GET /api/metrics HTTP/1.1
x-admin-token: your_admin_token_here
```

### Setting Up Admin Access

1. **Generate token** (see Environment Setup above)
2. **Store securely**: 
   - Add to Vercel env vars as `ADMIN_AUTH_TOKEN`
   - Add to server `.env` file
   - Share only with admin users
3. **Access admin endpoints**:
   - From monitoring dashboard: Uses token in HTTP client
   - Via curl: `curl -H "Authorization: Bearer TOKEN" https://your-domain.com/api/metrics`

### Example: Querying Admin Endpoints

```bash
# Get system metrics
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://your-domain.com/api/metrics | jq .

# Get active alerts
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://your-domain.com/api/alerts | jq .

# Acknowledge an alert
curl -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"alertId": "alert-uuid", "notes": "Resolved in deployment"}' \
  https://your-domain.com/api/alerts/acknowledge
```

---

## Monitoring & Observability

### Real-Time Monitoring

The app includes a built-in monitoring service that tracks:

- **Response Quality**: Average response quality scores (0-100)
- **Performance**: Average response time in milliseconds
- **Error Rates**: API errors as percentage of total requests
- **Crisis Detections**: Count of crisis patterns detected
- **User Satisfaction**: Rolling satisfaction rate

### View System Metrics

Access the metrics endpoint:

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://your-domain.com/api/metrics
```

Returns:

```json
{
  "success": true,
  "metrics": {
    "timestamp": "2026-04-08T22:45:00.000Z",
    "avg_response_quality": 78,
    "avg_response_time_ms": 1250,
    "total_requests": 4523,
    "error_count": 89,
    "error_rate": 1.97,
    "crisis_detections": 34,
    "regenerations": 156,
    "user_satisfaction_rate": 82.5
  }
}
```

### View Active Alerts

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://your-domain.com/api/alerts
```

Returns:

```json
{
  "success": true,
  "alerts": [
    {
      "id": "alert-123",
      "alert_type": "quality_decline",
      "severity": "warning",
      "message": "Response quality dropped to 45",
      "created_at": "2026-04-08T22:30:00.000Z",
      "acknowledged": false
    }
  ],
  "count": 1
}
```

### Alert Types

The system monitors for:

- **`crisis_escalation`** (severity: critical/warning)
  - Triggered when crisis patterns are detected
  - Requires immediate review

- **`quality_decline`** (severity: warning)
  - Triggered when response quality drops below 60
  - Indicates potential model or system issues

- **`api_error`** (severity: warning)
  - Triggered when API calls fail
  - Includes error type and message

- **`pattern_detected`** (severity: info)
  - Triggered on notable patterns (e.g., high regenerations)
  - Informational for trend analysis

- **`user_support_needed`** (severity: warning)
  - Triggered when user satisfaction drops below 50%
  - Indicates users may need better support

### Acknowledge Alerts

```bash
curl -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "alertId": "alert-123",
    "notes": "Issue resolved in v2.1.0"
  }' \
  https://your-domain.com/api/alerts/acknowledge
```

### Application Logging

The app uses structured logging with `pino`. Logs include:

- Timestamp
- Log level (debug, info, warn, error)
- Message
- Context (relevant data)

**Accessing logs:**

- **Vercel**: Dashboard → Deployments → Logs (real-time streaming)
- **Self-hosted**: Check PM2 logs:
  ```bash
  pm2 logs therapist
  ```

---

## Performance & Scaling

### Response Time Targets

- **Ideal**: < 2 seconds
- **Good**: 2-4 seconds
- **Acceptable**: 4-8 seconds
- **Degraded**: > 8 seconds

### Optimization Strategies

#### 1. Cache Strategy

```typescript
// Server-side caching for repeated user/session reads
const userCache = new Map();
const sessionCache = new Map();

// Clear caches on update
function invalidateCache(userId: string) {
  userCache.delete(userId);
}
```

#### 2. Connection Pooling

Supabase connections are automatically pooled. Monitor connection count:

```bash
# Supabase dashboard → Logs → Connections
# Should stay < 100 connections in normal operation
```

#### 3. Database Query Optimization

Indexes are created by migrations on:
- `profiles(user_id)`
- `chat_sessions(user_id, created_at)`
- `messages(session_id, created_at)`
- `daily_checkins(user_id, check_date)`
- `system_alerts(acknowledged, created_at)`

#### 4. API Rate Limiting

The app includes lightweight rate limiting:

```typescript
// Built-in rate limiter for abuse prevention
const rateLimit = {
  maxRequestsPerMinute: 60,
  maxChatsPerDay: 100,
};
```

### Scaling for High Load

#### Problem: Response Times > 4 seconds

**Solutions:**
1. Upgrade Supabase plan (adds more connections)
2. Enable Supabase connection pooling mode
3. Implement Redis for session caching (advanced)
4. Enable Vercel Edge Functions for routing (advanced)

#### Problem: OpenRouter Rate Limits

**Solutions:**
1. Upgrade OpenRouter plan
2. Implement request queuing (advanced)
3. Use fallback models for high load periods (configured in `lib/orchestrator.ts`)

#### Problem: High Database Costs

**Solutions:**
1. Archive old chat sessions (retention policy)
2. Aggregate metrics to reduce table size
3. Enable Supabase automatic backups with deletion

---

## Troubleshooting

### Issue: "Authentication Required" on Admin Endpoints

**Cause**: Missing or invalid `ADMIN_AUTH_TOKEN`

**Fix**:
```bash
# 1. Verify token in Vercel env vars
# 2. Check header is correctly formatted:
curl -H "Authorization: Bearer $ADMIN_TOKEN" https://your-domain.com/api/metrics

# 3. Generate new token if needed:
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### Issue: Supabase Connection Fails

**Cause**: Missing or incorrect env vars

**Fix**:
```bash
# 1. Verify in production:
# Vercel Settings → Environment Variables
# NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY present

# 2. Test connection:
curl https://your-domain.com/api/health
```

### Issue: Chat Responses Are Slow (> 8s)

**Cause**: Model timeout or network issue

**Check**:
1. View system metrics for error rate
2. Check Vercel logs for timeout errors
3. Verify OpenRouter account has credits

**Fix**:
1. Increase timeout if self-hosted: set `NODE_OPTIONS='--max_old_space_size=2048'`
2. Upgrade Vercel plan (gives more execution time)
3. Switch to faster model (in `OPENROUTER_MODEL` env var)

### Issue: "Response Feedback Already Submitted" Always Shows

**Cause**: localStorage not clearing or duplicate submission

**Fix**:
1. Clear browser cache: DevTools → Application → Clear Storage
2. Check server logs for duplicate POST requests
3. Verify `ResponseFeedback.tsx` is using correct UUID key

### Issue: Admin Alerts Not Triggering

**Cause**: Supabase migration not applied

**Fix**:
1. Check Supabase has `system_alerts` table
2. Run migration 0003: `supabase/migrations/0003_feedback_system.sql`
3. Verify table exists: Supabase → Table Editor

### Issue: Metrics Endpoint Returns Empty

**Cause**: No requests have been made yet (metrics are empty on startup)

**Normal behavior**: Metrics populate as users interact with app

---

## Disaster Recovery

### Database Backup Strategy

#### Supabase Automatic Backups

Supabase Pro plan includes daily backups. Enable in dashboard:

1. Go to **Settings → Backups**
2. Enable automatic backups
3. Set retention: 30 days recommended

#### Manual Backups

For critical data:

```bash
# Export data from Supabase
pg_dump postgresql://user:password@host/database > backup.sql

# Restore from backup
psql postgresql://user:password@host/database < backup.sql
```

### Recovery Procedures

#### Scenario: Database Corrupted

1. **Stop the app**: Prevent new writes
2. **Restore from backup**:
   ```bash
   # In Supabase dashboard → Backups → Restore
   ```
3. **Verify data integrity**: Run spot checks
4. **Restart the app**: Resume normal operation

#### Scenario: Admin Token Compromised

1. **Generate new token** (see Admin Authentication)
2. **Update Vercel env vars**: Set `ADMIN_AUTH_TOKEN` to new value
3. **Redeploy**: New deployments will use new token
4. **Rotate stored credentials**: Update any scripts/monitoring using old token

#### Scenario: Complete Outage

1. **Check status**:
   ```bash
   # Supabase status page: status.supabase.com
   # Vercel status page: vercel.statuspage.io
   # OpenRouter status: check their dashboard
   ```

2. **If Supabase is down**: App still works locally (graceful degradation)
3. **If Vercel is down**: Use failover or self-hosted instance
4. **If OpenRouter is down**: Falls back to keyword-based responses

---

## Security Best Practices

### 1. Environment Variables

✅ **Do:**
- Store in Vercel dashboard or secure env manager
- Use different values for dev/staging/production
- Rotate admin tokens quarterly

❌ **Don't:**
- Commit `.env` files to git
- Share tokens via email or chat
- Use same token across environments

### 2. Supabase Security

✅ **Do:**
- Enable Row Level Security (enabled by default)
- Use RLS policies to isolate user data
- Enable Supabase audit logs
- Use service role key only on backend

❌ **Don't:**
- Use service role key in frontend code
- Disable Row Level Security
- Share Supabase URL with untrusted parties

### 3. API Security

✅ **Do:**
- Validate all inputs with Zod schemas
- Use HTTPS everywhere (enforced by Vercel)
- Implement rate limiting
- Log security-relevant events

❌ **Don't:**
- Accept unsanitized JSON
- Trust client-provided IDs without verification
- Expose error messages that leak system info

### 4. Crisis Handling

✅ **Do:**
- Monitor crisis alerts closely
- Have escalation procedures
- Provide hotline numbers
- Log crisis interactions

❌ **Don't:**
- Rely solely on AI for crisis response
- Delay escalation
- Promise guaranteed safety

---

## Testing in Production

### Smoke Tests

Run immediately after deployment:

```bash
# Test public pages load
curl https://your-domain.com/                # landing
curl https://your-domain.com/auth            # auth page

# Test auth flow (create test account)
# Visit https://your-domain.com/auth
# Enter test email, click link in console logs

# Test main app
# Visit https://your-domain.com/app/chat
# Send test message, verify response

# Test admin endpoints
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://your-domain.com/api/metrics
```

### Load Testing

Use a tool like `k6` to verify performance:

```javascript
import http from 'k6/http';

export const options = {
  stages: [
    { duration: '2m', target: 10 },
    { duration: '5m', target: 20 },
    { duration: '2m', target: 0 },
  ],
};

export default function () {
  http.get('https://your-domain.com/api/strategies/recommend?session_type=venting');
}
```

Run:

```bash
k6 run load-test.js
```

### Monitoring Checklist

After deployment, monitor for 24 hours:

- [ ] Error rate < 5%
- [ ] Response time average < 4 seconds
- [ ] No critical alerts
- [ ] Database connections stable
- [ ] OpenRouter quota not exceeded
- [ ] No unusual traffic patterns

---

## Rollback Procedures

### If New Deployment Has Issues

#### Vercel:

```
Dashboard → Deployments → Select Previous Deployment → Revert
```

Takes ~1 minute, automatic DNS update.

#### Self-Hosted:

```bash
# Revert to previous commit
git revert HEAD

# Rebuild and redeploy
npm run build
pm2 restart therapist
```

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/genZ-AI-therapist)
- **Status**: [Supabase Status](https://status.supabase.com), [Vercel Status](https://vercel.statuspage.io)
- **Docs**: [Main README](./README.md)

---

**Last Updated**: April 8, 2026  
**Version**: 1.0 Production Ready
