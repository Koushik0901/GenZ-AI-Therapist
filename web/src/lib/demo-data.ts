export const seedMessages = [
  {
    role: "assistant" as const,
    content:
      "Hey. You don’t need to sound polished in here. Tell me what kind of weather your head is making today.",
  },
  {
    role: "user" as const,
    content:
      "I’m trying to do everything and somehow none of it feels done.",
  },
  {
    role: "assistant" as const,
    content:
      "That sounds like overload, not failure. Let’s shrink the problem until your body stops bracing against it.",
  },
];

export const journalEntries = [
  {
    id: "j-1",
    title: "Static Everywhere",
    mood: "Wired",
    createdAt: "Today, 8:12 AM",
    body: "My brain feels like seven browser tabs are all playing music. I need something smaller than a life plan.",
  },
  {
    id: "j-2",
    title: "Actually Proud",
    mood: "Steady",
    createdAt: "Yesterday, 9:40 PM",
    body: "I answered texts, made food, and didn’t spiral. That counts.",
  },
];

export const checkInSnapshot = {
  mood: 62,
  energy: 41,
  stress: 74,
  streak: 5,
};

export const insightCards = [
  {
    label: "Mood trend",
    value: "Climbing",
    detail: "Evenings are consistently calmer than your mid-afternoons.",
  },
  {
    label: "Recurring theme",
    value: "Overcommitment",
    detail: "Your lower-energy days often come after saying yes too much.",
  },
  {
    label: "Support win",
    value: "Naming the spiral sooner",
    detail: "You are catching overwhelm earlier instead of waiting for shutdown.",
  },
];

export const demoMemoryItems = [
  {
    id: "m-1",
    content: "Deadlines stack up fast when plans stay vague.",
    category: "stressors",
    status: "approved" as const,
    createdAt: "This week",
  },
  {
    id: "m-2",
    content: "Short, direct replies feel better than overly polished advice.",
    category: "preferences",
    status: "pending" as const,
    createdAt: "Today",
  },
];
