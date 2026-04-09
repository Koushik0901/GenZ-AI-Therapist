# GenZ AI Therapist

🫶 **Emotional support, reflection, and resource guidance built for Gen Z.**

**Status**: ✅ **v1.0.0 Production Ready** | [Deployment](./DEPLOYMENT.md) | [API Reference](./API_REFERENCE.md) | [Testing](./TESTING.md) | [Agentic Architecture](./AGENTIC-SYSTEM-OVERVIEW.md)

---

## ⚠️ Important: What This Is NOT

**This is not a replacement for therapy or professional mental health support.**

I built this to be a **supportive listening space** - not a substitute for real help. Here's what matters:

- I'm not a licensed therapist. I can't diagnose anything.
- I can't replace therapy, medication, or professional treatment.
- If someone is in immediate danger, thinking about self-harm, or in crisis, **they need real-world emergency support right now** - 988 (Suicide & Crisis Lifeline), Crisis Text Line (text HELLO to 741741), or calling 911.
- I can help people untangle their thoughts, notice patterns, and feel heard. But I'm not trained to treat mental health conditions.

This app is meant for **reflection, support, and guidance** - like talking to a thoughtful friend who actually listens. That's valuable. But it's not therapy.

---

## What I Built

I created an AI emotional support app that actually understands what someone needs and adapts to help them better.

Most chatbots just throw your message at an LLM and hope for a good response. I built something smarter:

- **It understands context.** When someone vents, I know they need validation, not solutions. When they ask for help, I know they want actionable steps.
- **It validates its own work.** If the response isn't warm or validating enough, it regenerates until it gets it right.
- **It learns about each user.** If someone responds well to short answers or certain kinds of support, the app remembers that and adapts.
- **It detects when something's wrong.** Over multiple messages, I can notice if someone's mood is declining or if a crisis is escalating.
- **It's safe by design.** Multi-layer crisis detection, not just keyword matching.

The app helps people:

- vent without needing a perfect opener
- untangle messy thoughts
- track mood, energy, stress
- notice patterns over time
- journal, chat, and remember all in one place
- get pointed toward actual resources that help

---

## How It Works (The Smart Part)

When someone messages "I've been crying all day and nothing helps," here's what actually happens behind the scenes:

**First, I understand the situation:**
- What are they really saying? Venting or looking for solutions?
- How are they emotionally? I infer their mood, energy, stress levels.
- Are they safe? I check for subtle crisis signs, not just obvious ones.

**Then I decide what they need:**
- This person is venting - they need listening and validation.
- Resources might actually feel dismissive right now, so I skip them.
- Adjust my tone and approach specifically for this kind of conversation.

**Then I generate a response - but I don't just send it:**
- I grade my own response on 4 dimensions: warmth, validation, clarity, relevance.
- If it scores below passing (65), I don't send it - I regenerate it with a specific fix.
- If warmth is low, I make it warmer. If validation is low, I emphasize feelings more.

**For longer conversations, I detect patterns:**
- Is mood trending down over multiple messages?
- Are they repeating the same topic obsessively?
- Is a crisis escalating? I flag this for human follow-up.

**And I learn continuously:**
- User feedback: "This helped!" → That strategy gets better for them.
- Comments like "too long" → I remember they prefer short.
- I test different approaches and see what actually works.

This is why it feels more thoughtful than a generic chatbot.

---

## 🎉 What I Built in v1.0.0

This is a **full production release** of the agentic system:

- ✅ **Intelligent AI Orchestrator** - 6 phases, 13 specialized decision-makers working together
- ✅ **Multi-Layer Crisis Detection** - Explicit keywords, implicit patterns, wellness signals, escalation tracking
- ✅ **User Learning System** - Learns what works for each person and adapts automatically
- ✅ **Self-Validating Responses** - Grades its own work and regenerates if not good enough
- ✅ **Pattern Detection** - Notices wellness trends, escalation, repeated topics over time
- ✅ **Production Security** - Authentication, input validation, data isolation, crisis handling
- ✅ **Comprehensive Testing** - 166 tests, 89% passing, all critical systems verified
- ✅ **Full Documentation** - Deployment guide, API reference, architecture breakdown, testing results
- ✅ **Graceful Degradation** - Works even if some services fail

See [CHANGELOG.md](./CHANGELOG.md) for what changed and [AGENTIC-SYSTEM-OVERVIEW.md](./AGENTIC-SYSTEM-OVERVIEW.md) to understand how the intelligent system works.

---

## ✨ What This App Does

I built this as a **supportive listening space** with multiple ways to connect:

The whole experience is designed to feel:

- warm, not sterile
- supportive, not preachy
- safe, not fake-clinical
- real, not robotic

**If your brain is being loud, spirally, flat, or overloaded, this app gives you a space to talk, reflect, understand the vibe, and move one step forward.**

---

## 💬 The Main Chat ("Yap")

This is where the intelligence I built really shows up.

When you send a message, it doesn't just go to the LLM and pray. Here's what actually happens:

1. **I understand what you're saying**
   - What's your sentiment? (positive, negative, crisis)
   - What do you actually need? (venting, advice, validation, information)
   - How are you emotionally? (mood, energy, stress levels)

2. **I decide what you need**
   - If you're venting: validate feelings, listen, skip advice
   - If you want help: offer structure and steps
   - If you need validation: affirm and normalize
   - If you're in crisis: emergency protocols first

3. **I generate a response tailored to you**
   - Tone adjusted to what helps you
   - Length based on your preferences (some people want short, some want detailed)
   - Resources only when they actually help

4. **I validate my own work**
   - Is it warm enough? Does it really validate your feelings?
   - Is it clear? Easy to understand?
   - Does it match what you need?
   - If not, I regenerate until it's good.

5. **I remember patterns**
   - After a few messages, I start noticing trends
   - Is your mood declining? I track it.
   - Are you repeating the same worry? I notice.
   - Is a crisis building? I flag it.

The result: **Chat that feels like someone actually listened and understood what you need.**

### 📓 Journal Studio

Sometimes your thoughts are too big for chat bubbles. This is where you write it all out.

You can:
- Write a full journal entry with a title
- Log your mood with it
- Save to your account and revisit anytime
- Actually have space to think without interruption

Because some feelings need a page, not a text exchange.

### 🌡️ Daily Vibe Check

Quick 30-second self-report: How are you really doing?

Log:
- mood
- energy
- stress
- a short note

Why this matters: **You know yourself better than any AI.** Your actual self-report is more trustworthy than me guessing. I use this to understand real patterns.

### 📈 Pattern Tea

This is where I show you what the data is actually saying.

Over time, the app turns your check-ins and conversations into patterns you can see:

- "my stress has been high for days and I keep pretending it's fine"
- "my energy always crashes halfway through the week"
- "I think I'm okay, but the pattern says... actually no"

You get:
- Trend charts showing how you're changing
- Weekly averages
- Plain-language insight cards

The goal isn't to judge you. It's to make invisible patterns visible so you can actually understand what's happening.

### 🧩 Memory & Settings

You control what I remember about you.

- See what the app learned about you
- Approve the useful stuff
- Hide what feels creepy
- Reset whenever you want

Memory should feel helpful, not invasive.

### 🔐 Sign In & Your Data

You sign in with a magic link (just your email, no passwords).

That means:
- Your chat history stays yours
- Journal entries are private
- Your vibe checks aren't shared
- Only you can see your data

Everything is truly yours, not shared with anyone else or used to train AI models.

---

## 🧭 How I Built This (Core Principles)

### 🩺 Not therapy. Not a replacement.

I'm clear about what I am:

**I'm built for:**
- Emotional support and listening
- Reflection and untangling thoughts
- Noticing patterns over time
- Grounding and perspective
- Pointing toward helpful resources

**I'm NOT built for:**
- Diagnosis or clinical assessment
- Treatment or medication advice
- Emergency response
- Replacing therapy or professional help

If someone needs actual treatment or professional support, they should get it. I can help them feel heard while they do.

### 🛡️ Safe by design

I hardened the system around real risks:

- **Multi-layer crisis detection** - Explicit keywords, implicit patterns, wellness signals, escalation tracking. Not just keyword matching.
- **Blocked jailbreak attempts** - People won't trick me into breaking my own rules.
- **Trusted resource filtering** - If I suggest resources, they're actually vetted and helpful.
- **Refuses harmful requests** - But still routes legitimate help-seeking into safety.

### 🔒 Your data is actually yours

Using Supabase Row Level Security means:
- Your chats are completely private to you
- Nobody else (not even me as an admin without proper auth) can see another user's data
- Your journal, check-ins, memory - all private
- Your data doesn't get used to train future models

### 🎨 Design that supports, not alienates

The visual design isn't random styling:

- **Warm colors** = feeling supported, not sterile
- **Clear typography** = easy to read when your brain is overwhelmed
- **Spacious layout** = breathing room, not cramped
- **Accessible interactions** = works for everyone, not just tech people

The goal is an interface that feels like a safe space, not a corporate product.

---

## 🚀 Feature Breakdown

### 💬 Chat Features

- markdown-rendered assistant replies
- async-friendly chat save path
- first-message session creation
- generated chat titles
- sticky resource panel
- starter prompts before the first user message
- animated pending assistant bubble
- session archive + delete controls
- fixed-screen app layout with internal pane scrolling

### 🛡️ Safety + Prompt Engineering Features

- prompt-injection detection before the main agent flow
- system prompts that treat user/history/search content as untrusted data
- structured output validation with `zod`
- graceful fallback when model JSON is broken
- markdown-only contract for the final assistant reply
- crisis-aware support routing
- trusted-domain resource filtering

### 🌡️ Wellness + Insight Features

- manual vibe checks
- chat-inferred mood / energy / stress
- trend charts
- weekly averages
- streak snapshots
- plain-language insight cards

### 🧠 Memory Features

- per-item memory status
- `pending`, `approved`, `hidden`
- account-private memory controls

### ⚡ Performance + UX Features

- fixed-screen authenticated app shell
- route loading skeletons
- top loading bar during route transitions
- optimized package imports for heavier UI libraries
- request-scoped server caching for repeated viewer/session reads
- lightweight rate limiting for abuse prevention on free-tier hosting

---

## 🧱 The Tech Behind It

### Frontend
- **Next.js 16** - The actual app you use
- **React 19** - Making the UI work
- **Tailwind CSS** - Styling that's warm, not corporate
- **Recharts** - Charts for Pattern Tea

### Backend & Database
- **Supabase** - Where your data lives (PostgreSQL + authentication)
- **Next.js API routes** - The server logic

### The AI System
- **OpenRouter** - Access to different AI models
- **TypeScript orchestrator** - The 6-phase intelligent system I built
- **Zod** - Making sure structured data is actually structured

**Want to understand how the AI orchestrator works?** Check out [AGENTIC-SYSTEM-OVERVIEW.md](./AGENTIC-SYSTEM-OVERVIEW.md) - it explains the 13 tools and 6-phase decision system.

The code lives in:
- `web/src/lib/orchestrator.ts` - The 6-phase coordinator
- `web/src/lib/tools/` - The 13 specialized decision-makers
- `web/src/app/api/` - The API endpoints

### Note on the old code
There's still Python code in the repo from when this started as a Streamlit prototype. But the **real product** is the Next.js app in `web/`. The Python stuff is historical context.

---

## 🗂️ Project Structure (What You Need to Know)

```
web/                          ← The actual product
├─ src/app/                   ← Pages and API routes
├─ src/components/            ← UI components
├─ src/lib/                   ← Business logic
│  ├─ orchestrator.ts         ← The 6-phase AI system
│  └─ tools/                  ← 13 specialized tools
└─ package.json
```

That's the current product. There's old Python code in the repo (from a prototype), but ignore it - this is what actually runs.

---

## 🧭 The App Pages

**Public:**
- `/` - Landing page
- `/auth` - Sign in with email

**After signing in:**
- `/app/chat` - Yap (main chat)
- `/app/journal` - Journal Studio
- `/app/check-in` - Daily Vibe Check
- `/app/insights` - Pattern Tea
- `/app/settings` - Memory & privacy controls

---

## 🔑 Getting It Running

### What You Need

```dotenv
# These are required
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=minimax/minimax-m2.5
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

That's it. Put these in a `.env` file in the root directory (or `web/.env.local`).

---

## 🚀 Getting Started

### 1. Set Up Supabase (5 minutes)

Go to [supabase.com](https://supabase.com), create a free project, grab the URL and key.

In the Supabase settings:
- **Site URL**: `http://localhost:3000`
- **Redirect URL**: `http://localhost:3000/auth/callback`

Then run the database setup in Supabase's SQL editor:
```sql
-- Copy the contents of supabase/migrations/0001_init.sql and run it
```

That creates all the tables and sets up security.

### 2. Get Your Keys

You need:
- Supabase URL + Anon Key (from Supabase dashboard)
- OpenRouter API Key (from [openrouter.ai](https://openrouter.ai))

### 3. Run Locally

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`

### 4. To Deploy

Follow [DEPLOYMENT-CHECKLIST.md](./DEPLOYMENT-CHECKLIST.md) - takes 45 minutes, gets you live.

---

## How to Use It

**Sign in first** (`/auth`):
- Enter your email
- Click the magic link in your inbox
- You're in

**Then explore:**
- **Yap** (`/app/chat`) - Main chat for support, venting, perspective
- **Journal** (`/app/journal`) - Write when you need space to think
- **Check-in** (`/app/check-in`) - Quick daily mood/energy/stress log
- **Insights** (`/app/insights`) - See patterns the app notices
- **Settings** (`/app/settings`) - Control memory and privacy

---

## 🌍 Deploying to Production

I designed this to run on **free tiers** of Vercel + Supabase (for now).

**Setup:**
- **Frontend + API**: Vercel (hosts the Next.js app)
- **Database + Auth**: Supabase (PostgreSQL + login)
- **AI Model**: OpenRouter (pay-per-use)

**To deploy:**
1. Push your code to GitHub
2. Connect GitHub to Vercel
3. Set environment variables in Vercel dashboard
4. Done - it auto-deploys

For step-by-step: see [DEPLOYMENT-CHECKLIST.md](./DEPLOYMENT-CHECKLIST.md)

---

## 🔐 Security & Privacy

**I'm serious about protecting your data:**

- Every user is isolated. Your data is only visible to you.
- Supabase Row Level Security means I can't accidentally give someone else your data.
- No data is sold, shared, or used to train models.

**Against attacks:**
- I block jailbreak attempts and prompt injection
- Input validation everywhere
- Careful with what I trust from external sources

**Crisis safety:**
- This app can support and point you toward help
- But it's **not** emergency care
- If you're in immediate danger, call 911 or text 988 (Suicide & Crisis Lifeline)

---

## 📚 Want to Learn More?

- [AGENTIC-SYSTEM-OVERVIEW.md](./AGENTIC-SYSTEM-OVERVIEW.md) - How the AI system actually works
- [DEPLOYMENT-CHECKLIST.md](./DEPLOYMENT-CHECKLIST.md) - Step-by-step to deploy
- [API_REFERENCE.md](./API_REFERENCE.md) - The backend endpoints
- [TESTING.md](./TESTING.md) - Test results and how to run them

---

## A Final Note

**I built this with care, but it's not a replacement for real help.**

If you're struggling with mental health, please reach out to:
- **988** - Suicide & Crisis Lifeline (call or text)
- **Text HELLO to 741741** - Crisis Text Line
- A therapist, counselor, or trusted person in your life

This app is a supportive space. Use it. But don't use it instead of getting actual help.

Take care of yourself. 🫶
