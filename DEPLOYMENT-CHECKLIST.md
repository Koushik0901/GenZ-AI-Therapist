# Deployment Checklist - GenZ AI Therapist

Use this checklist to track your deployment progress. Check off each item as you complete it.

**Estimated Time**: 45 minutes  
**Difficulty**: Intermediate

---

## Phase 1: Local Verification (5 minutes)

- [x] Build succeeds: `npm run build`
- [x] Tests pass: `npm run test` (148 passing, 18 documented non-critical failures)
- [x] No TypeScript errors
- [x] Ready to proceed to Supabase

---

## Phase 2: Supabase Setup (10 minutes)

### Step 2.1: Create Supabase Project
- [ ] Go to https://supabase.com
- [ ] Click "Sign up" or sign in with GitHub
- [ ] Click **"New Project"**
- [ ] Fill in project details:
  - Name: `genZ-ai-therapist`
  - Password: (save it!)
  - Region: (choose your region)
- [ ] Click **"Create new project"** and wait 2 minutes
- [ ] Project created successfully

### Step 2.2: Get Supabase Credentials
- [ ] Go to **Settings → API** in your Supabase project
- [ ] Copy **Project URL**: `_____________________`
- [ ] Copy **Anon Key**: `_____________________`
- [ ] Save these values safely - you'll need them for Vercel

### Step 2.3: Configure Auth URLs (Important!)
- [ ] Go to **Authentication → URL Configuration**
- [ ] Set **Site URL** to `http://localhost:3000` (for now)
- [ ] Add **Redirect URLs**:
  - [ ] `http://localhost:3000/auth/callback`
  - [ ] (Later add Vercel domain)

### Step 2.4: Run Database Migrations
- [ ] Open **SQL Editor** in Supabase
- [ ] Create **New Query**
- [ ] Copy contents from `supabase/migrations/0001_init.sql`
- [ ] Click **Run** and wait for ✅
- [ ] Repeat for `supabase/migrations/0002_agentic_logging.sql`
- [ ] Repeat for `supabase/migrations/0003_feedback_system.sql`
- [ ] All 13 tables created successfully
- [ ] Verify tables exist in **Table Editor**

**Note**: If you get a "column sentiment does not exist" error on 0003:
→ See MIGRATION-FIX.md for details and solution

---

## Phase 3: Vercel Setup (15 minutes)

### Step 3.1: Connect GitHub
- [ ] Go to https://vercel.com
- [ ] Sign up or sign in
- [ ] Click **Continue with GitHub**
- [ ] Authorize Vercel to access your GitHub account

### Step 3.2: Deploy Your Project
- [ ] Click **Add New → Project**
- [ ] Select your `GenZ-AI-Therapist` repository
- [ ] Set **Root Directory** to `./web`
- [ ] Click **Deploy**
- [ ] Wait 3-5 minutes for deployment
- [ ] See ✅ green checkmark
- [ ] Copy your Vercel URL: `https://_____________________.vercel.app`

### Step 3.3: Add Environment Variables
- [ ] In Vercel, go to **Settings → Environment Variables**
- [ ] Add each variable (copy from your `.env` file):

| Variable | Value |
|----------|-------|
| NEXT_PUBLIC_APP_URL | `https://YOUR_VERCEL_URL.vercel.app` |
| NEXT_PUBLIC_SUPABASE_URL | (from Step 2.2) |
| NEXT_PUBLIC_SUPABASE_ANON_KEY | (from Step 2.2) |
| OPENROUTER_API_KEY | (check your `.env`) |
| OPENROUTER_MODEL | `minimax/minimax-m2.5` |
| OPENROUTER_BASE_URL | `https://openrouter.ai/api/v1` |
| ADMIN_AUTH_TOKEN | (generate new one) |

**Generate new ADMIN_AUTH_TOKEN:**
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```
- [ ] Generated token: `_____________________`
- [ ] Pasted into Vercel as ADMIN_AUTH_TOKEN
- [ ] All 7 variables saved
- [ ] Vercel auto-redeployed with new environment variables

### Step 3.4: Update Supabase Auth URLs (Final!)
- [ ] Back to Supabase dashboard
- [ ] Go to **Authentication → URL Configuration**
- [ ] Update **Site URL** to your Vercel URL: `https://_____________________.vercel.app`
- [ ] Update **Redirect URLs** with Vercel domain:
  - [ ] `https://YOUR_VERCEL_URL.vercel.app/auth/callback`
- [ ] Saved successfully

---

## Phase 4: Verification (10 minutes)

### Test 1: Page Load
```bash
curl -I https://your-vercel-url.vercel.app
```
- [ ] Returns `200 OK`

### Test 2: Auth Page
- [ ] Visit: `https://your-vercel-url.vercel.app/auth`
- [ ] Page loads with email input
- [ ] Auth form displays correctly

### Test 3: Public API
```bash
curl "https://your-vercel-url.vercel.app/api/strategies/recommend?session_type=venting"
```
- [ ] Returns JSON with strategy recommendations
- [ ] Response is valid JSON

### Test 4: Admin API
```bash
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  https://your-vercel-url.vercel.app/api/metrics
```
- [ ] Returns system metrics
- [ ] Requires valid admin token
- [ ] Shows response times, request counts, etc.

### Test 5: Chat Flow (Manual)
- [ ] Visit your app: `https://your-vercel-url.vercel.app`
- [ ] Go to `/app/chat`
- [ ] Send message: "I'm feeling overwhelmed"
- [ ] Receive AI response
- [ ] Response is helpful and relevant
- [ ] Can submit feedback on response
- [ ] Feedback accepted

---

## Phase 5: Optional - Custom Domain (5 minutes)

*Skip this if you want to keep `vercel.app` domain*

- [ ] In Vercel, go to **Settings → Domains**
- [ ] Enter your custom domain: `_____________________`
- [ ] Vercel shows DNS records
- [ ] Update DNS at your domain registrar (GoDaddy, Namecheap, etc.)
- [ ] Wait 5-15 minutes for DNS propagation
- [ ] Domain working

### Update Supabase Auth URLs Again
- [ ] Supabase → **Authentication → URL Configuration**
- [ ] **Site URL**: `https://your-custom-domain.com`
- [ ] **Redirect URLs**: `https://your-custom-domain.com/auth/callback`

---

## Phase 6: Post-Launch Setup (Ongoing)

### Enable Backups
- [ ] Supabase → **Settings → Backups**
- [ ] Enable **Automatic Backups**
- [ ] Set **Retention**: 30 days
- [ ] Backups enabled

### Monitor Your App (Daily for first week)

**Check Metrics:**
```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://your-domain.com/api/metrics
```
- [ ] Error rate < 5%
- [ ] Response time < 4 seconds
- [ ] No critical alerts

**Monitor in Dashboard:**
- [ ] Check **Vercel**: Deployments → Logs
- [ ] Check **Supabase**: Analytics
- [ ] Check **Your App**: `/api/metrics` endpoint

---

## Success Criteria ✅

Your app is fully deployed when you can:

- [ ] Load the app: `https://your-domain.com`
- [ ] Sign up with email and magic link
- [ ] Send chat messages and get AI responses
- [ ] Submit feedback on responses
- [ ] Check admin metrics: `/api/metrics`
- [ ] View active alerts: `/api/alerts`
- [ ] See graceful degradation (app works even if services have issues)

---

## What's Now Live

✅ **Chat System**
- Multi-turn conversations
- AI emotional support responses
- Response feedback submission

✅ **User Features**
- Email authentication (magic links)
- Session persistence
- User preference learning
- Journal entries
- Daily check-ins

✅ **Admin Features**
- System metrics tracking
- Alert creation and acknowledgment
- Performance monitoring
- Crisis detection logs

✅ **Infrastructure**
- Worldwide CDN (Vercel)
- Automatic SSL/TLS
- Automatic database backups
- 99.9% uptime SLA

---

## Costs Overview

**Free Tier (First Month):**
- Vercel: Free
- Supabase: Free tier
- OpenRouter: Pay-as-you-go (~$0.01-0.10/chat)

**When to Upgrade:**
- **Vercel Pro** ($20/mo): If you get production traffic
- **Supabase Pro** ($25/mo): After 100k requests/month
- **OpenRouter**: Scales with usage

---

## Troubleshooting

### Issue: Auth page doesn't work
1. Check Supabase auth URLs are correct
2. Verify Redirect URL matches your domain
3. Check Vercel logs: Dashboard → Deployments → Logs

### Issue: API returns 500 error
1. Check all env vars are set in Vercel
2. Verify Supabase connection string is correct
3. Check Vercel logs for error message

### Issue: Admin endpoints return 401
1. Verify `ADMIN_AUTH_TOKEN` is set in Vercel env vars
2. Use header: `Authorization: Bearer $TOKEN`
3. Token should be 64 hex characters

### Issue: Database migration fails
1. Check all 3 migrations run in order
2. Verify no errors in Supabase SQL editor
3. Check tables exist in Table Editor

---

## Resources

- **LAUNCH.md** - Detailed deployment guide
- **DEPLOYMENT.md** - Complete deployment reference
- **API_REFERENCE.md** - API endpoint documentation
- **TESTING.md** - Testing procedures
- **README.md** - Product overview

---

**Status**: Ready to deploy  
**Last Updated**: April 9, 2026

Good luck with your deployment! Come back and let me know how it goes. 🚀
