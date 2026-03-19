# GenZ AI Therapist Web App

This is the **current main product app** for GenZ AI Therapist.

It uses:

- **Next.js 16 App Router** for the web app
- **Supabase** for auth + database
- **OpenRouter** for model access
- **LangGraph for JavaScript** for the agentic workflow

## Agentic Workflow

The live app does **not** use CrewAI.

The current workflow is built with **LangGraph JS** and lives in:

- [`src/lib/companion-graph.ts`](./src/lib/companion-graph.ts)
- [`src/lib/companion-nodes.ts`](./src/lib/companion-nodes.ts)
- [`src/lib/companion-foundation.ts`](./src/lib/companion-foundation.ts)

The flow includes:

1. prompt-injection detection
2. guard / routing
3. title generation
4. sentiment + intent classification
5. mood / energy / stress inference
6. resource discovery + filtering
7. final non-clinical response generation

## Run Locally

```bash
npm install
npm run dev
```

Then open:

- `http://localhost:3000`

## Notes

- Magic-link auth needs Supabase configured
- AI replies need OpenRouter configured
- Live resource search needs `SERPER_API_KEY`

For the full product docs, setup guide, and repo history, see the root [`README.md`](../README.md).
