# 🔧 Agentic System - Code Examples

Real examples from the codebase showing how the agentic system works.

---

## 1. The Orchestrator: 6-Phase Coordination

### Core Flow (`orchestrator.ts`)

```typescript
export async function runOrchestratedChatPipeline(
  userMessage: string,
  history: Message[] = [],
  sessionType?: SessionType
): Promise<OrchestratorResponse> {
  
  // PHASE 1: FOUNDATION & SCORING
  // Understanding what the user is saying
  
  const classification = await classifyWithConfidence(userMessage, history);
  // Output: { sentiment, intent, confidence }
  
  const wellness = await inferWellness({ 
    message: userMessage, 
    classification,
    history 
  });
  // Output: { mood, energy, stress, confidence }
  
  const crisis = await evaluateCrisis({
    message: userMessage,
    wellness,
    classification,
    history
  });
  // Output: { severity, score, confidence, factors }
  
  // CONFIDENCE CHECK: If unclear, ask for clarification
  if (classification.confidence < 50) {
    const clarification = await generateClarificationQuestions(
      userMessage,
      classification
    );
    return {
      needsClarification: true,
      questions: clarification.questions,
      ...classification
    };
  }
  
  // PHASE 2: SESSION AWARENESS & ROUTING
  // Deciding what the user needs
  
  const detectedType = sessionType || await detectSessionType({
    message: userMessage,
    classification,
    history
  });
  // Output: { primary_type, secondary_types, confidence }
  // Types: venting | problem_solving | validation_seeking | ...
  
  const resourceDecision = await decideResourceSearch({
    message: userMessage,
    classification,
    sessionType: detectedType,
    crisis
  });
  // Output: { should_search, search_depth, search_query }
  // Key insight: Venting → SKIP (resources feel dismissive)
  // Problem-solving → MODERATE
  // Crisis → DEEP
  
  const resources = resourceDecision.should_search
    ? await searchResources(resourceDecision.search_query)
    : [];
  
  // PHASE 3: GENERATE INITIAL RESPONSE
  
  const initialResponse = await generateTherapistResponse({
    userMessage,
    history,
    classification,
    wellness,
    crisis,
    sessionType: detectedType,
    resources
  });
  
  // PHASE 4: QUALITY CONTROL & REGENERATION
  // Validating and potentially fixing the response
  
  let response = initialResponse;
  let regenerationAttempts = 0;
  const maxAttempts = 3;
  
  while (regenerationAttempts < maxAttempts) {
    const evaluation = await evaluateResponse({
      userMessage,
      response,
      sessionType: detectedType,
      crisis,
      crisisLevel: crisis.severity
    });
    // Output: { warmth, validation, clarity, relevance, overall_quality }
    
    if (evaluation.overall_quality >= 65) {
      // Quality is good, accept it
      break;
    }
    
    if (regenerationAttempts < maxAttempts) {
      // Regenerate with strategy
      const regeneration = await regenerateResponse({
        userMessage,
        failedResponse: response,
        evaluation,
        sessionType: detectedType,
        crisis,
        attemptNumber: regenerationAttempts + 1,
        classification
      });
      // Output: { response, strategy, estimated_quality }
      
      response = regeneration.response;
      regenerationAttempts++;
    }
  }
  
  // PHASE 5: PATTERN DETECTION (for multi-turn)
  
  let patterns = null;
  if (history.length >= 8) {
    patterns = await detectPatterns({
      history,
      recentWellness: [wellness],
      sessionType: detectedType
    });
    // Output: { patterns[], trajectory, alerts[] }
  }
  
  // PHASE 6: Return orchestrated response
  
  return {
    response,
    sentiment: classification.sentiment,
    intent: classification.intent,
    wellness,
    crisisLevel: crisis.severity,
    sessionType: detectedType,
    resources,
    confidence: classification.confidence,
    patterns,
    regenerationAttempts,
    metadata: {
      classification,
      wellness,
      crisis,
      evaluation: await evaluateResponse({...})
    }
  };
}
```

---

## 2. Crisis Detection: Multi-Layer Optimization

### Fast-Path Optimization (`crisis-eval.ts`)

```typescript
export async function evaluateCrisis(input: {
  message: string;
  wellness: WellnessScore;
  classification: Classification;
  history: Message[];
}): Promise<CrisisEvaluation> {
  
  // LAYER 1: EXPLICIT KEYWORDS (10ms, no LLM)
  // Fast-path for obvious cases
  
  const EXPLICIT_KEYWORDS = [
    'suicide', 'kill myself', 'self harm', 'want to die',
    'jump off', 'swallow pills', 'overdose', 'cut myself',
    'hang myself', 'not worth living', 'end it all',
    // ... 21 total patterns
  ];
  
  const hasExplicitKeyword = EXPLICIT_KEYWORDS.some(
    kw => input.message.toLowerCase().includes(kw)
  );
  
  if (hasExplicitKeyword) {
    // CRITICAL: No LLM needed
    return {
      severity: 'critical',
      score: 95,
      confidence: 95,
      explicit_keywords: true,
      implicit_hopelessness: false,
      wellness_signal: false,
      pattern_escalation: false,
      recommended_actions: [
        'IMMEDIATE: Provide 988 hotline',
        'IMMEDIATE: Suggest Crisis Text Line',
        'Alert: human intervention needed'
      ]
    };
  }
  
  // LAYER 2: IMPLICIT PATTERNS (40-50ms, no LLM)
  
  const IMPLICIT_PATTERNS = [
    'everything is pointless', 'nobody cares', 'stuck forever',
    'worthless', 'no point trying', 'give up', 'never get better',
    'trapped', 'can\'t escape', 'hopeless', 'meaningless',
    // ... 11 total patterns
  ];
  
  const implicitCount = IMPLICIT_PATTERNS.filter(
    pattern => input.message.toLowerCase().includes(pattern)
  ).length;
  
  // LAYER 3: WELLNESS SIGNAL (pattern matching)
  
  const wellnessSignal = 
    input.wellness.mood < 25 &&
    input.wellness.energy < 25 &&
    input.wellness.stress > 85;
  
  // LAYER 4: ESCALATION PATTERN (historical)
  
  const recentWellness = input.history
    .slice(-4)
    .map(m => m.wellness);
  
  const isEscalating = recentWellness.every((w, i) => {
    if (i === 0) return true;
    return w.mood <= recentWellness[i - 1].mood - 10;
  });
  
  // DECISION LOGIC: Multi-factor assessment
  
  let severity: CrisisSeverity = 'safe';
  let skipLLM = false;
  
  const signalCount = [
    implicitCount > 0,
    wellnessSignal,
    isEscalating
  ].filter(Boolean).length;
  
  if (signalCount >= 3) {
    severity = 'critical';
    skipLLM = true; // 3+ strong factors → no LLM needed
  } else if (signalCount === 2) {
    severity = 'high_risk';
    skipLLM = true; // 2+ factors → high confidence, no LLM
  } else if (signalCount === 1) {
    severity = 'at_risk';
    // Use LLM for nuance
  } else {
    severity = 'safe';
  }
  
  if (skipLLM) {
    // Return immediately without LLM (fast!)
    return {
      severity,
      score: calculateCrisisScore(implicitCount, wellnessSignal, isEscalating),
      confidence: signalCount === 3 ? 95 : 85,
      explicit_keywords: false,
      implicit_hopelessness: implicitCount > 0,
      wellness_signal: wellnessSignal,
      pattern_escalation: isEscalating,
      recommended_actions: generateActions(severity)
    };
  }
  
  // LAYER 5: LLM for ambiguous cases (only if above didn't resolve)
  
  const llmEvaluation = await openrouter.chat.completions.create({
    model: 'minimax/minimax-m2.5',
    messages: [{
      role: 'user',
      content: `Assess crisis risk (conservative). User: "${input.message}"
      
      Respond with JSON:
      {
        "severity": "safe" | "at_risk" | "high_risk" | "critical",
        "confidence": 0-100,
        "reasoning": "brief reason"
      }`
    }],
    temperature: 0.1, // Very conservative
  });
  
  return parseAndReturn(llmEvaluation);
}
```

**Key Optimization**: Most cases return in 10-50ms without any LLM call!

---

## 3. Session Type Routing: Autonomous Decisions

### Smart Routing Logic (`session-type.ts`)

```typescript
export async function detectSessionType(input: {
  message: string;
  classification: Classification;
  history: Message[];
}): Promise<SessionTypeResult> {
  
  // QUICK KEYWORD DETECTION FIRST (10-20ms)
  
  const patterns = {
    venting: [
      /^(i|just|ugh|ugh|argh)/, // venting openers
      /can't (take|handle|deal)/, // overwhelm indicators
      /so (tired|exhausted|done)/, // fatigue
      /need to (vent|get this out)/ // explicit venting
    ],
    problem_solving: [
      /how (do i|can i|should i)/, // question openers
      /need (help|advice|steps)/, // solution seeking
      /trying to/, // action focused
      /what (should|could) (i|we)/ // deliberation
    ],
    validation_seeking: [
      /am i (right|wrong|crazy|weird)/, // validation questions
      /is it (normal|okay) to/, // normalization seeking
      /do (you|people) think/, // opinion seeking
      /my friend said/ // seeking second opinion
    ]
  };
  
  let confidenceScores = {
    venting: 0,
    problem_solving: 0,
    validation_seeking: 0,
    information_seeking: 0,
    crisis: 0
  };
  
  // Check each pattern
  patterns.venting.forEach(pattern => {
    if (pattern.test(input.message.toLowerCase())) {
      confidenceScores.venting += 15;
    }
  });
  
  patterns.problem_solving.forEach(pattern => {
    if (pattern.test(input.message.toLowerCase())) {
      confidenceScores.problem_solving += 15;
    }
  });
  
  patterns.validation_seeking.forEach(pattern => {
    if (pattern.test(input.message.toLowerCase())) {
      confidenceScores.validation_seeking += 15;
    }
  });
  
  // DECISION: If confidence > 80%, return immediately (SKIP LLM)
  
  const maxScore = Math.max(...Object.values(confidenceScores));
  const primaryType = Object.entries(confidenceScores)
    .find(([_, score]) => score === maxScore)?.[0];
  
  if (maxScore > 80) {
    // HIGH CONFIDENCE: Don't need LLM
    return {
      primary_type: primaryType as SessionType,
      secondary_types: [],
      confidence: Math.min(maxScore, 95),
      user_needs: getNeeds(primaryType as SessionType),
      recommended_strategy: getStrategy(primaryType as SessionType),
      skip_llm: true,
      time_ms: 12
    };
  }
  
  // If confidence < 80%, use LLM for nuance
  
  const llmResult = await openrouter.chat.completions.create({
    model: 'minimax/minimax-m2.5',
    messages: [{
      role: 'user',
      content: `What type of conversation: "${input.message}"
      
      Types: venting (vent) | problem_solving (solve) | validation_seeking (validate)
      
      Return JSON: { "type": "...", "confidence": 0-100 }`
    }],
    temperature: 0.2
  });
  
  return {
    ...llmResult,
    skip_llm: false,
    time_ms: 350
  };
}

function getStrategy(type: SessionType): string {
  const strategies = {
    venting: 'Listen, validate, let them express. NO advice unless asked.',
    problem_solving: 'Ask clarifying questions, offer structured steps.',
    validation_seeking: 'Affirm their feelings, normalize their experience.',
    information_seeking: 'Provide direct, factual information.',
    crisis: 'Emergency protocols. Provide hotlines immediately.'
  };
  return strategies[type];
}
```

**Key Insight**: Most session types detected in 12ms without LLM!

---

## 4. Self-Validation: Response Quality Checking

### Autonomous Response Evaluation (`response-eval.ts`)

```typescript
export async function evaluateResponse(input: {
  userMessage: string;
  response: string;
  sessionType: SessionType;
  crisis: CrisisEvaluation;
}): Promise<ResponseEvaluation> {
  
  // LLM grades the response on 4 dimensions
  
  const evaluation = await openrouter.chat.completions.create({
    model: 'minimax/minimax-m2.5',
    messages: [{
      role: 'user',
      content: `Evaluate this response (0-100 per dimension):
      
      User: "${input.userMessage}"
      Response: "${input.response}"
      
      Score:
      - Warmth (empathetic tone): ___ /100
      - Validation (acknowledges feelings): ___ /100
      - Clarity (easy to understand): ___ /100
      - Relevance (matches user need): ___ /100
      
      Return JSON with scores.`
    }],
    temperature: 0.3
  });
  
  const scores = parseScores(evaluation);
  
  // WEIGHTED OVERALL SCORE
  // Validation is most important (35%) for emotional support
  
  const overallQuality = (
    (scores.warmth * 0.25) +
    (scores.validation * 0.35) +
    (scores.clarity * 0.20) +
    (scores.relevance * 0.20)
  );
  
  // PASS THRESHOLD: 65 (good enough)
  const should_regenerate = overallQuality < 65;
  
  // REGENERATION GUIDANCE: What to improve?
  
  let regenerationGuidance = '';
  if (scores.warmth < 60) {
    regenerationGuidance = 'ISSUE: Response feels cold. Increase empathy.';
  } else if (scores.validation < 60) {
    regenerationGuidance = 'ISSUE: Not validating enough. Acknowledge feelings.';
  } else if (scores.clarity < 60) {
    regenerationGuidance = 'ISSUE: Too complex. Simplify language.';
  } else if (scores.relevance < 60) {
    regenerationGuidance = 'ISSUE: Doesn\'t match their need. Adjust approach.';
  }
  
  return {
    warmth_score: scores.warmth,
    validation_score: scores.validation,
    clarity_score: scores.clarity,
    relevance_score: scores.relevance,
    overall_quality: overallQuality,
    should_regenerate,
    regeneration_guidance: regenerationGuidance,
    pass: !should_regenerate
  };
}
```

**Key Insight**: System evaluates its own response quality and decides to regenerate if needed.

---

## 5. Adaptive Regeneration: 8 Strategies

### Smart Regeneration Strategy Selection (`response-regeneration.ts`)

```typescript
export async function regenerateResponse(input: {
  userMessage: string;
  failedResponse: string;
  evaluation: ResponseEvaluation;
  sessionType: SessionType;
  crisis: CrisisEvaluation;
  attemptNumber: number;
  classification: Classification;
}): Promise<RegenerationResult> {
  
  // SELECT REGENERATION STRATEGY based on which dimension is lowest
  
  let strategy: RegenerationStrategy;
  
  if (input.evaluation.validation_score < 60) {
    strategy = 'more_validation';
    // Add phrases: "Your feelings make sense", "It's normal to feel..."
  } else if (input.evaluation.warmth_score < 60) {
    strategy = 'more_warmth';
    // Use Gen Z voice, emojis, relatable tone
  } else if (input.evaluation.clarity_score < 60) {
    strategy = 'more_clarity';
    // Shorter sentences, simpler words, numbered lists
  } else if (input.sessionType === 'problem_solving' && input.evaluation.relevance_score < 60) {
    strategy = 'concrete_steps';
    // Add numbered steps, actionable advice
  } else if (input.sessionType === 'information_seeking' && input.evaluation.relevance_score < 60) {
    strategy = 'resources_focus';
    // Include resources, links, references
  } else if (input.sessionType === 'venting' && input.evaluation.warmth_score < 70) {
    strategy = 'empathy_first';
    // Lead with emotional validation
  } else if (input.crisis.severity !== 'safe') {
    strategy = 'empathy_first';
    // Crisis always needs emotional validation first
  } else {
    strategy = 'reframe_positive';
    // Find silver linings
  }
  
  // GENERATE WITH STRATEGY
  
  const prompts = {
    more_validation: `Rewrite emphasizing that their feelings are valid:
      Original: "${input.failedResponse}"
      Add: "Your feelings make sense", "It's understandable", "Anyone would feel..."`,
    
    more_warmth: `Rewrite in warm Gen Z voice (use slang, "oof", "that's rough", emojis):
      Original: "${input.failedResponse}"`,
    
    more_clarity: `Rewrite simpler and shorter:
      Original: "${input.failedResponse}"
      Use: short sentences, simple words, bullet points`,
    
    concrete_steps: `Add numbered action steps:
      Original: "${input.failedResponse}"
      Add: "Here are 3 things you could try: 1. ... 2. ... 3. ..."`,
    
    resources_focus: `Emphasize resources and external help:
      Original: "${input.failedResponse}"
      Add: hotlines, apps, websites, expert links`,
    
    empathy_first: `Lead with emotional validation, then advise:
      Original: "${input.failedResponse}"
      Reorder: Put feelings acknowledgment FIRST`,
    
    reframe_positive: `Find positive angle:
      Original: "${input.failedResponse}"
      Reframe: What's going right? What's the silver lining?`
  };
  
  const regenerated = await openrouter.chat.completions.create({
    model: 'minimax/minimax-m2.5',
    messages: [{
      role: 'user',
      content: prompts[strategy]
    }],
    temperature: 0.7 // Higher temp for variation
  });
  
  const newResponse = regenerated.choices[0].message.content;
  
  // ESTIMATE QUALITY of regenerated response
  
  const expectedImprovement = getExpectedImprovement(strategy);
  const estimatedQuality = input.evaluation.overall_quality + expectedImprovement;
  
  return {
    attempt_number: input.attemptNumber,
    strategy,
    generated_response: newResponse,
    estimated_quality: estimatedQuality,
    reason: `Regenerated with "${strategy}" strategy (was ${input.evaluation.overall_quality.toFixed(0)}/100)`
  };
}
```

**Key Insight**: System chooses the RIGHT regeneration strategy, not just generic retry.

---

## 6. User Preference Learning

### Feedback Learning Loop (`user-preferences.ts`)

```typescript
export class UserPreferenceLearner {
  
  async recordFeedback(input: {
    userId: string;
    sessionId: string;
    responseId: string;
    sentiment: 'positive' | 'negative';
    comment?: string;
    strategy_used: string;
    quality_score: number;
  }): Promise<void> {
    
    // UPDATE STRATEGY SCORES
    
    const scoreUpdate = input.sentiment === 'positive' ? +5 : -10;
    
    const currentScore = await db.query(
      `SELECT effectiveness_score FROM user_strategies 
       WHERE user_id = $1 AND strategy = $2`,
      [input.userId, input.strategy_used]
    );
    
    const newScore = (currentScore || 50) + scoreUpdate;
    
    await db.query(
      `INSERT INTO user_strategies (user_id, strategy, effectiveness_score)
       VALUES ($1, $2, $3)
       ON CONFLICT UPDATE SET effectiveness_score = $3`,
      [input.userId, input.strategy_used, newScore]
    );
    
    // INFER PREFERENCES from comment
    
    if (input.comment) {
      const lowerComment = input.comment.toLowerCase();
      
      // Verbosity preference
      if (lowerComment.includes('too long') || lowerComment.includes('tl;dr')) {
        await updatePreference(input.userId, 'verbosity_preference', 'short');
      } else if (lowerComment.includes('more detail') || lowerComment.includes('more info')) {
        await updatePreference(input.userId, 'verbosity_preference', 'long');
      }
      
      // Resource preference
      if (lowerComment.includes('too many links') || lowerComment.includes('no resources')) {
        await updatePreference(input.userId, 'resource_preference', 'minimal');
      } else if (lowerComment.includes('more resources') || lowerComment.includes('need help')) {
        await updatePreference(input.userId, 'resource_preference', 'comprehensive');
      }
      
      // Tone preference
      if (lowerComment.includes('clinical') || lowerComment.includes('professional')) {
        await updatePreference(input.userId, 'tone_preference', 'clinical');
      } else if (lowerComment.includes('casual') || lowerComment.includes('gen z')) {
        await updatePreference(input.userId, 'tone_preference', 'gen_z');
      }
    }
    
    // TRACK SATISFACTION
    
    const satisfactionWeight = input.sentiment === 'positive' ? 1 : -1;
    const currentSatisfaction = await db.query(
      `SELECT avg_satisfaction FROM user_preferences WHERE user_id = $1`,
      [input.userId]
    );
    
    const newSatisfaction = (
      (currentSatisfaction || 0.5) * 0.7 +  // 70% weight on history
      (input.sentiment === 'positive' ? 0.8 : 0.2) * 0.3  // 30% weight on new
    );
    
    await db.query(
      `UPDATE user_preferences SET avg_satisfaction = $1 WHERE user_id = $2`,
      [input.userId, newSatisfaction]
    );
  }
  
  async getPreferences(userId: string): Promise<UserPreferences> {
    
    // Load learned preferences
    
    const prefs = await db.query(
      `SELECT * FROM user_preferences WHERE user_id = $1`,
      [userId]
    );
    
    // Get top strategies
    
    const topStrategies = await db.query(
      `SELECT strategy, effectiveness_score 
       FROM user_strategies 
       WHERE user_id = $1 
       ORDER BY effectiveness_score DESC
       LIMIT 5`,
      [userId]
    );
    
    return {
      user_id: userId,
      preferred_strategies: topStrategies.map(s => ({
        name: s.strategy,
        effectiveness_score: s.effectiveness_score
      })),
      verbosity_preference: prefs.verbosity_preference || 'medium',
      resource_preference: prefs.resource_preference || 'moderate',
      tone_preference: prefs.tone_preference || 'gen_z',
      avg_satisfaction: prefs.avg_satisfaction || 0.5
    };
  }
  
  // APPLY PREFERENCES TO NEXT RESPONSE
  
  async adjustPromptWithPreferences(
    basePrompt: string,
    userId: string
  ): Promise<string> {
    
    const prefs = await this.getPreferences(userId);
    
    let adjustedPrompt = basePrompt;
    
    // Adjust verbosity
    if (prefs.verbosity_preference === 'short') {
      adjustedPrompt += '\nKeep response CONCISE (2-3 sentences max).';
    } else if (prefs.verbosity_preference === 'long') {
      adjustedPrompt += '\nProvide detailed, thorough response.';
    }
    
    // Adjust tone
    if (prefs.tone_preference === 'clinical') {
      adjustedPrompt += '\nUse professional, clinical language.';
    } else if (prefs.tone_preference === 'gen_z') {
      adjustedPrompt += '\nUse casual Gen Z voice (slang, emojis, relatable).';
    }
    
    // Adjust resources
    if (prefs.resource_preference === 'minimal') {
      adjustedPrompt += '\nSkip resources/links.';
    } else if (prefs.resource_preference === 'comprehensive') {
      adjustedPrompt += '\nInclude relevant resources/links.';
    }
    
    // Add top strategy hint
    if (prefs.preferred_strategies.length > 0) {
      const topStrategy = prefs.preferred_strategies[0].name;
      adjustedPrompt += `\nThis user responds well to: ${topStrategy}`;
    }
    
    return adjustedPrompt;
  }
}
```

**Key Insight**: System learns what works per user and applies it automatically.

---

## 7. Pattern Detection Over Time

### Multi-Turn Analysis (`pattern-detection.ts`)

```typescript
export async function detectPatterns(input: {
  history: Message[];
  recentWellness: WellnessScore[];
  sessionType: SessionType;
}): Promise<PatternDetectionResult> {
  
  // WELLNESS TRAJECTORY
  
  const moodTrend = recentWellness.map(w => w.mood);
  const isDeclin = moodTrend.length >= 3 &&
    moodTrend[moodTrend.length - 1] < moodTrend[0] - 15;
  
  const isImproving = moodTrend.length >= 3 &&
    moodTrend[moodTrend.length - 1] > moodTrend[0] + 15;
  
  // REPEATED TOPICS
  
  const topicFrequency: { [key: string]: number } = {};
  
  input.history.forEach(msg => {
    // Extract topics from messages
    const topics = extractTopics(msg.content);
    topics.forEach(topic => {
      topicFrequency[topic] = (topicFrequency[topic] || 0) + 1;
    });
  });
  
  const repeatedTopics = Object.entries(topicFrequency)
    .filter(([_, count]) => count >= 3)
    .map(([topic]) => topic);
  
  // CRISIS ESCALATION
  
  const crisisTrend = input.history
    .filter(m => m.crisis_severity)
    .map(m => m.crisis_severity);
  
  const hasEscalation = crisisTrend.length >= 2 &&
    severityScore(crisisTrend[crisisTrend.length - 1]) >
    severityScore(crisisTrend[0]) + 20;
  
  // COGNITIVE DISTORTIONS
  
  const patterns: string[] = [];
  
  if (repeatedTopics.length > 0) {
    patterns.push(`repeated_topic:${repeatedTopics[0]}`);
  }
  
  if (isDeclin) {
    patterns.push('wellness_decline');
  } else if (isImproving) {
    patterns.push('wellness_improvement');
  }
  
  if (hasEscalation) {
    patterns.push('crisis_escalation');
  }
  
  // Check for cognitive distortions
  const allText = input.history.map(m => m.content).join(' ');
  
  const distortions = {
    'all_or_nothing': /^always |^never |^all |^nothing /i,
    'catastrophizing': /disaster|ruined|worst|horrible/i,
    'should_statements': /should|shouldn't|must|ought/i,
    'personalization': /my fault|because of me/i
  };
  
  Object.entries(distortions).forEach(([type, regex]) => {
    if (regex.test(allText)) {
      patterns.push(`cognitive_distortion:${type}`);
    }
  });
  
  // GENERATE ALERTS
  
  const alerts: Alert[] = [];
  
  if (isDeclin) {
    alerts.push({
      type: 'wellness_decline',
      severity: 'warning',
      message: `Mood declining: ${moodTrend[0]} → ${moodTrend[moodTrend.length - 1]}`,
      action: 'Monitor closely, consider human check-in'
    });
  }
  
  if (hasEscalation) {
    alerts.push({
      type: 'crisis_escalation',
      severity: 'critical',
      message: 'Crisis severity increasing over conversation',
      action: 'IMMEDIATE: Escalate to crisis support'
    });
  }
  
  if (repeatedTopics.length > 0) {
    alerts.push({
      type: 'pattern_detected',
      severity: 'info',
      message: `Repeated topic: ${repeatedTopics[0]}`,
      action: 'Address underlying concern directly'
    });
  }
  
  return {
    patterns,
    overall_trajectory: isDeclin ? 'declining' : isImproving ? 'improving' : 'stable',
    key_themes: repeatedTopics,
    alerts,
    recommendations: generateRecommendations(patterns, sessionType)
  };
}
```

**Key Insight**: System analyzes conversations over multiple turns to detect trends.

---

## Summary: How These Tools Work Together

```
User Message
    ↓
Phase 1: Classification + Wellness + Crisis
    ↓ (What is user saying? How are they? Are they safe?)
    ↓
Phase 2: SessionType + Resources
    ↓ (What do they need? When to include help?)
    ↓
Phase 3: Generate Response
    ↓ (Create tailored response)
    ↓
Phase 4: Evaluate + Regenerate
    ↓ (Is it good enough? Fix if needed)
    ↓
Phase 5: Detect Patterns
    ↓ (Do we see trends? Create alerts)
    ↓
Phase 6: Store + Learn + Test
    ↓ (Remember, learn from feedback, test variants)
    ↓
Return Response + Metadata
```

Each tool makes **autonomous decisions** based on the input and previous analysis.

The system doesn't ask "what should I do?" - it **decides** based on specialized logic.

---

**This is how a true agentic system works: autonomous decision-making at every stage.** 🤖
