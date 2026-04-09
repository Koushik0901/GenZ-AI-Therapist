# GenZ AI Therapist

🫶 **Hi. I'm here to listen.**

**Status**: ✅ **v1.0.0 Production Ready** | [Deployment](./DEPLOYMENT.md) | [API Reference](./API_REFERENCE.md) | [Testing](./TESTING.md) | [How I Work](./AGENTIC-SYSTEM-OVERVIEW.md)

---

## ⚠️ Important: What I Am & What I'm Not

**I'm not a therapist. I can't replace real help.**

I'm a thoughtful listening space—like a friend who really hears you. That matters. But:
- I can't diagnose, prescribe, or treat anything
- I'm not therapy. Real therapy is with a trained person
- I'm not emergency care. If you're in immediate danger: **988**, **text HELLO to 741741**, or **911**

**Use me alongside real help, not instead of it.**

---

## What I Do

When you message me:

1. **I listen and understand** - Not just keywords. I know if you're venting (need validation), problem-solving (need steps), seeking validation (need affirmation), or in crisis (need resources).

2. **I check if you're safe** - Multi-layer crisis detection: explicit keywords, implicit patterns, wellness signals.

3. **I decide what you need** - Adjust my tone, length, and approach based on what helps.

4. **I validate my own work** - I score myself (warmth, validation, clarity, relevance). If I'm not good enough (below 65%), I rewrite myself.

5. **I learn about you** - Over conversations, I notice what works for you and adapt.

6. **I notice patterns** - Declining mood? Repeated worries? Crisis building? I flag it.

This is why conversations feel more thoughtful than generic chatbots.

---

## What I Can & Can't Do

✅ Listen without judgment  
✅ Help untangle messy thoughts  
✅ Notice patterns over time  
✅ Offer grounding and perspective  
✅ Adapt to what works for you  

❌ Diagnose anything  
❌ Treat conditions  
❌ Replace therapy or medication  
❌ Be emergency care  

---

## How to Use Me

**💬 Yap** - Main chat. Vent, ramble, ask for support. I'm most thoughtful here.

**📓 Journal** - For thoughts too big for chat. You write, I listen.

**🌡️ Check-in** - Daily pulse. You know yourself best. Tell me your mood/energy/stress.

**📈 Insights** - I show patterns I've noticed over time.

**🧩 Settings** - You control what I remember. Approve, hide, or reset anytime.

---

## Built On

- **6 phases of intelligent routing** - Not just LLM-ing your message
- **13 specialized decision-makers** - Each handles something specific
- **Multi-layer crisis detection** - 10ms keywords → 50ms patterns → AI only when needed
- **Self-validating responses** - I grade and improve my own work
- **Continuous learning** - Feedback teaches me what works for each person
- **Your actual privacy** - Row-level security, no data selling, you control what I remember

For deep details: [How I Actually Work](./AGENTIC-SYSTEM-OVERVIEW.md)

---

## v1.0.0 Production Release

✅ Intelligent system (6 phases, 13 tools)  
✅ Safety first (multi-layer crisis detection)  
✅ User learning (adapts to what works for you)  
✅ Self-validating (grades and improves responses)  
✅ Pattern recognition (notices trends)  
✅ Tested (166 tests, 89% passing)  
✅ Production ready (auth, security, isolation)  
✅ Documented (deployment, API, architecture)  
✅ Resilient (works even if services fail)  

---

## Getting Started

### Local Setup (5 minutes)

1. Create a Supabase project (free tier works) → grab your URL and anon key from Settings → API

2. Get an OpenRouter key at [openrouter.ai](https://openrouter.ai) (free account)

3. Set environment variables:
```dotenv
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXT_PUBLIC_SUPABASE_URL=your_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_key
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=minimax/minimax-m2.5
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

4. Run:
```bash
cd web && npm install && npm run dev
```

Open `http://localhost:3000`

### Using It

- Sign in at `/auth`
- Start in **Yap** (`/app/chat`) - I'm most thoughtful here
- Use other features as needed

### Going Live (45 min)

Follow [DEPLOYMENT-CHECKLIST.md](./DEPLOYMENT-CHECKLIST.md) to deploy to Vercel + Supabase

---

## Learn More

- [How I Actually Work](./AGENTIC-SYSTEM-OVERVIEW.md) - 6-phase system, 13 tools, detailed architecture
- [Deployment Guide](./DEPLOYMENT-CHECKLIST.md) - Step-by-step to go live
- [API Reference](./API_REFERENCE.md) - All endpoints
- [Testing Results](./TESTING.md) - What's been verified
- [Version History](./CHANGELOG.md) - What's new in v1.0.0

---

## Get Real Help

If you're struggling with mental health:

- **988** - Suicide & Crisis Lifeline (call or text)
- **Crisis Text Line** - Text HELLO to 741741
- A therapist, counselor, or someone you trust

I'm honest about my limits because I care about you getting actual help.

Take care of yourself. 🫶
