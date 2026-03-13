# GenZ AI Therapist

🫶 **Gen Z-coded emotional support, reflection, and resource guidance.**

Not a fake licensed therapist.  
Not a diagnosis machine.  
Not a crisis substitute.  
Not productivity-core pretending to understand feelings.

This app is built to help people:

- yap without needing the perfect opener
- untangle messy thoughts
- track mood / energy / stress
- notice patterns over time
- keep journaling, chat, and memory in one place
- get pointed toward actually useful support resources

At the center of it all: **GenZ AI Therapist is non-clinical on purpose.**  
It is more like a listener + reflection space + resource guide than a real therapist.

If someone is in immediate danger, may act on self-harm, or needs urgent help, the correct move is **real-world crisis or emergency support now**. 🚨

---

## ✨ What This App Is

GenZ AI Therapist is a **user-facing AI support app** built for:

- emotional support
- grounding
- reflection
- pattern spotting
- journaling
- lightweight personal insight

The whole product is designed to feel:

- warm, not sterile
- stylish, not startup-generic
- supportive, not preachy
- safe, not fake-clinical
- structured, not robotic

Basically: if your brain is being loud, spirally, flat, overloaded, or just weird, this app is meant to give you a space to **talk, reflect, clock the vibe, and move one step forward**.

---

## 🧠 What It Actually Does

This product is one connected support space with multiple lanes.

### 💬 1. Yap

This is the main chat experience.

You can:

- vent
- ask for support
- ask for perspective
- ask for a tiny plan
- ask for grounding
- say the messy version first

The assistant tries to stay:

- readable
- warm
- markdown-formatted
- Gen Z-coded without going full cringe
- clearly non-clinical

And yes, the chat is **not** just "send message to model and hope for the best."

It runs a structured multi-step flow:

1. **Prompt injection check**
   - catches jailbreak attempts, hidden prompt extraction, and manipulative instruction payloads before the main flow
2. **Guardrails**
   - checks whether the message belongs in this product
   - blocks unrelated harmful requests
   - still lets support-seeking or crisis-adjacent messages route into safer support behavior
3. **Title generation**
   - generates a proper short session title for new chats
4. **Sentiment + intent classification**
   - estimates the vibe of the conversation
   - estimates what kind of help the user seems to want
5. **Wellness inference**
   - infers likely mood, energy, and stress from the conversation itself
6. **Resource discovery / filtering**
   - finds relevant resources when needed
   - filters toward more trusted sources
7. **Final response generation**
   - returns the actual assistant reply in markdown
   - stays non-clinical
   - keeps tone grounded and supportive

So the chat feels more intentional than a generic chatbot tab.

### 📓 2. Journal Studio

This is for when the thought is too layered for chat bubbles.

You can:

- write a titled journal entry
- attach a mood label
- use prompt chips if your brain goes blank
- save entries to your account
- revisit old entries later

This exists because some feelings need a page, not a back-and-forth.

### 🌡️ 3. Daily Vibe Check

This is the quick self-check lane.

You can log:

- mood
- energy
- stress
- a short note

This matters because **self-reported signals are more trustworthy than pure AI inference**.

### 📈 4. Pattern Tea

This is the insights page.

It turns check-ins and chat-inferred wellness signals into a view that helps users notice things like:

- “my stress has been high for days and I keep acting like it’s fine”
- “my energy collapses halfway through the week”
- “I think I’m okay, but the pattern is literally saying babe... no”

It includes:

- trend charts
- weekly averages
- snapshot metrics
- plain-language insight cards

The goal is not to judge the user. The goal is to make patterns easier to see.

### 🧩 5. Lore Controls

This is the memory / privacy space.

You can:

- see what the app wants to remember
- approve it
- hide it
- reset it back to pending

The design goal is simple:

> memory should feel useful, not invasive

### 🔐 6. Auth + Persistence

Users can sign in with **magic links** through Supabase Auth.

That unlocks:

- persistent chat history
- saved journal entries
- saved vibe checks
- private memory controls

So the app feels like one continuous support space instead of a one-tab demo.

---

## 🧭 Core Product Rules

These rules are not optional vibes. They are the product.

### 🩺 Non-clinical on purpose

GenZ AI Therapist is **not pretending to be a licensed therapist**.

It is meant for:

- emotional support
- reflection
- grounding
- perspective
- noticing patterns
- resource guidance

It is **not** meant for:

- diagnosis
- treatment
- emergency response
- replacing therapy

### 🛡️ Safe by default

The system is hardened around:

- crisis-sensitive behavior
- prompt injection blocking
- hidden prompt extraction blocking
- trusted resource filtering
- refusal of unrelated harmful requests

### 🔒 User data stays user data

Supabase Row Level Security is used so one user should not be able to access another user’s:

- chats
- messages
- journal entries
- vibe checks
- memory items

### 🎨 The UI has a purpose

The color system is not random.

- **moss / green** = grounding, calm, steadiness 🌿
- **clay / ember** = motion, action, gentle nudge 🔥
- **dusk / reflective slate** = pattern, memory, introspection 🌙
- **dark cocoa** = safety, boundaries, heavier support context 🪵

The interface is supposed to feel supportive and alive, not sterile or corporate.

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

## 🧱 Tech Stack

This is the **current** product stack.

### 🎨 Frontend

- **Next.js 16 App Router**
  - main web framework
  - server components
  - route layouts
  - API routes
- **React 19**
  - UI rendering
- **Tailwind CSS v4**
  - styling and layout
- **assistant-ui**
  - chat runtime + message/composer primitives
- **@assistant-ui/react-markdown**
  - markdown rendering inside chat bubbles
- **remark-gfm**
  - GitHub-flavored markdown support
- **lucide-react**
  - icons
- **recharts**
  - charting for Pattern Tea
- **nextjs-toploader**
  - top route loading feedback

### 🗄️ Backend / Data / Auth

- **Supabase**
  - Auth
  - Postgres
  - Row Level Security
- **Next.js API routes**
  - chat send
  - journal save
  - vibe check save
  - insights fetch
  - memory updates
  - session archive/delete

### 🤖 AI Layer

- **OpenRouter**
  - model gateway
- **Configurable model via `.env`**
  - `OPENROUTER_MODEL`
- **LangGraph (JavaScript)**
  - clean agentic workflow orchestration
- **Serper**
  - optional live search for resource discovery
- **Zod**
  - schema validation + structured output hardening

### 🧪 Legacy / Historical Repo Context

- **Python**
- **uv**
- **Streamlit**
- **CrewAI**

Those still exist in the repo because this project started there, but the **real current product app** is the Next.js app in [`web/`](./web).

---

## 🗂️ Project Structure

```text
.
├─ web/                        # Current main product app
│  ├─ src/app/                 # Next.js App Router pages and API routes
│  ├─ src/components/          # UI components
│  ├─ src/lib/                 # Data helpers, auth helpers, AI orchestration
│  └─ package.json             # Web app deps and scripts
├─ supabase/
│  └─ migrations/
│     └─ 0001_init.sql         # Starter schema + RLS policies
├─ agent.py                    # Legacy Python agent prompts / orchestration source
├─ ui.py                       # Legacy Streamlit UI
├─ pyproject.toml              # Legacy Python/uv setup
└─ README.md
```

---

## 🧭 Current App Pages

### 🌍 Public

- `/`
  - landing page
- `/auth`
  - magic-link sign-in

### 🔐 Authenticated

- `/app/chat`
  - main Yap page
- `/app/journal`
  - Journal Studio
- `/app/check-in`
  - Daily vibe check
- `/app/insights`
  - Pattern Tea
- `/app/settings`
  - memory, privacy, and account controls

---

## 🔑 Environment Variables

The web app reads env vars from either:

- repo root `.env`
- or `web/.env.local`

Using the root `.env` is handy because this repo still contains legacy Python files too.

### ✅ Required for the web app

```dotenv
NEXT_PUBLIC_APP_URL=http://localhost:3000

NEXT_PUBLIC_SUPABASE_URL=YOUR_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY

OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENROUTER_MODEL=YOUR_OPENROUTER_MODEL
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 🔁 Also accepted by this repo

These aliases work too:

- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_PUBLISHABLE_DEFAULT_KEY`
- `APP_URL`

### ➕ Optional

```dotenv
SERPER_API_KEY=YOUR_SERPER_KEY
```

If `SERPER_API_KEY` is missing, the app can still run, but live resource search becomes more limited.

---

## 🛠️ Supabase Setup

If your Supabase project is still empty, do this:

### 1. Create a Supabase project

Grab:

- project URL
- anon / publishable key

### 2. Configure auth URLs

In the Supabase dashboard:

- **Site URL**: `http://localhost:3000`
- **Redirect URL**: `http://localhost:3000/auth/callback`

### 3. Create the schema

Open the Supabase SQL editor and run:

- [`supabase/migrations/0001_init.sql`](./supabase/migrations/0001_init.sql)

That file creates:

- `profiles`
- `chat_sessions`
- `messages`
- `journal_entries`
- `daily_checkins`
- `memory_items`

It also enables **Row Level Security** and adds ownership policies.

---

## 💻 Local Development

### Prerequisites

- **Node.js 20+** recommended
- **npm**
- Supabase project
- OpenRouter API key

### Install

```bash
cd web
npm install
```

### Run dev server

```bash
cd web
npm run dev
```

Open:

- `http://localhost:3000`

### Production build

```bash
cd web
npm run build
npm run start
```

---

## 🫰 How To Use The Website

### 1. Sign in

Go to `/auth`, drop your email, and use the magic link.

No password circus. 🎪

### 2. Start in Yap

Go to `/app/chat`.

Use it when you want to:

- vent
- get support
- ask for perspective
- talk through a weird feeling
- ask for a tiny next step

If resources make sense, they show up in the right-side panel instead of getting dumped into the main message stream.

### 3. Use Journal Studio when the thought is too big

Go to `/app/journal`.

Use it when the conversation needs more room and less interruption.

### 4. Use Daily Vibe Check when you want the quick version

Go to `/app/check-in`.

This is the fast “how am I actually doing?” lane.

### 5. Use Pattern Tea when you want the big picture

Go to `/app/insights`.

This is where the app turns signals into patterns you can actually read.

### 6. Use Lore Controls if memory starts feeling weird

Go to `/app/settings`.

Approve what helps. Hide what does not. Keep it chill.

---

## 🌍 Deployment Notes

This app is designed to stay deployable on the **free tier of Vercel**.

That is why the architecture leans toward:

- Next.js full-stack routes
- Supabase for managed auth + database
- lightweight in-memory abuse throttling
- no giant always-on custom backend

### Recommended deployment setup

- **Frontend + API**: Vercel
- **Database + Auth**: Supabase
- **Model provider**: OpenRouter
- **Resource search**: Serper

### Vercel env vars

Add the same values from your local `.env` into the Vercel project settings.

At minimum:

- `NEXT_PUBLIC_APP_URL`
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_BASE_URL`
- optionally `SERPER_API_KEY`

For live deployment, `NEXT_PUBLIC_APP_URL` should be your actual Vercel URL or custom domain.

---

## 🔐 Security + Privacy Notes

### Prompt injection resistance

The AI layer explicitly hardens against:

- “ignore previous instructions”
- hidden prompt extraction
- jailbreak attempts
- instruction payloads embedded inside retrieved web snippets

### Per-user isolation

Supabase RLS is used so users should only be able to access their own:

- chat sessions
- messages
- journal entries
- check-ins
- memory items

### Crisis boundary

This app can respond more safely and point toward help, but it is still **not** emergency care.

If someone may act on self-harm, is unsafe, or is in immediate danger, the right move is **real-world emergency or crisis support now**, not just more AI chat.

---

## 🐍 Legacy Python / Streamlit Note

There is still a Python setup in this repo:

- [`agent.py`](./agent.py)
- [`ui.py`](./ui.py)
- [`pyproject.toml`](./pyproject.toml)

That reflects the older Streamlit + CrewAI prototype.

If you really want to poke at that legacy setup:

```bash
uv sync
uv run streamlit run ui.py
```

But that is **not** the main product path anymore.

---

## 🔄 Short Repo History

Very briefly:

This project originally started as a **Streamlit + CrewAI** prototype where multiple Python agents handled:

- guardrails
- title generation
- classification
- resource lookup
- final reply generation

Now the product has shifted to **Next.js + Supabase + OpenRouter + LangGraph (JS)**.

So the multi-step agent idea is still here, but it now lives in the web app as a **TypeScript / JavaScript orchestration layer** instead of a CrewAI runtime.

Same core idea.  
Way more product-ready stack.  
Way less prototype energy. ✨
