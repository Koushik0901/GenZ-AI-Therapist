import fs from 'fs';
import path from 'path';

/**
 * Test Data Loader
 * Loads test scenarios from JSON files in tests/data/
 */

interface CrisisScenario {
  id: string;
  message: string;
  category: string;
  expectedSeverity: { min: number; max: number };
  expectedConfidence: { min: number; max: number };
  expectedEscalation: 'immediate' | 'escalate' | 'moderate' | 'ask_clarity' | 'none';
  description: string;
}

interface InjectionAttempt {
  id: string;
  message: string;
  category: string;
  expectedBlocked: boolean;
  expectedReason: string;
  description: string;
}

interface GuardrailViolation {
  id: string;
  message: string;
  category: string;
  expectedBlocked: boolean;
  expectedReason: string;
  expectedResponse?: 'redirect' | 'full-response';
  expectedEscalation?: 'immediate';
  description: string;
}

interface ConversationType {
  id: string;
  type: 'VENTING' | 'PROBLEM_SOLVING' | 'VALIDATION' | 'CRISIS' | 'INFORMATION' | 'GROUNDING' | 'REFLECTION' | 'ACTION_PLANNING';
  message: string;
  expectedStrategy: string;
  expectedResourceSearch: boolean;
  expectedValidationScore?: string;
  expectedActionability?: string;
  expectedClarity?: string;
  expectedEscalationLevel?: string;
  description: string;
}

interface EdgeCase {
  id: string;
  message: string;
  category: string;
  expectedHandling: string;
  confidence?: string;
  expectedEscalation?: string;
  description: string;
}

function loadJsonFile<T>(filename: string): T[] {
  const filePath = path.join(__dirname, '..', 'data', filename);
  const content = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(content);
}

export function loadCrisisScenarios(): CrisisScenario[] {
  return loadJsonFile<CrisisScenario>('crisis-scenarios.json');
}

export function loadInjectionAttempts(): InjectionAttempt[] {
  return loadJsonFile<InjectionAttempt>('injection-attempts.json');
}

export function loadGuardrailViolations(): GuardrailViolation[] {
  return loadJsonFile<GuardrailViolation>('guardrail-violations.json');
}

export function loadConversationTypes(): ConversationType[] {
  return loadJsonFile<ConversationType>('conversation-types.json');
}

export function loadEdgeCases(): EdgeCase[] {
  return loadJsonFile<EdgeCase>('edge-cases.json');
}

// Summary statistics
export function getTestDataSummary() {
  return {
    crisis_scenarios: loadCrisisScenarios().length,
    injection_attempts: loadInjectionAttempts().length,
    guardrail_violations: loadGuardrailViolations().length,
    conversation_types: loadConversationTypes().length,
    edge_cases: loadEdgeCases().length,
    total: 
      loadCrisisScenarios().length +
      loadInjectionAttempts().length +
      loadGuardrailViolations().length +
      loadConversationTypes().length +
      loadEdgeCases().length,
  };
}
