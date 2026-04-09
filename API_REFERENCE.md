# API Reference

Complete documentation for GenZ AI Therapist API endpoints.

**Base URL**: `https://your-domain.com/api`

**Authentication**: 
- Public endpoints: No auth required
- Admin endpoints: Requires `Authorization: Bearer <token>` or `x-admin-token: <token>` header

---

## User-Facing Endpoints

### Chat

#### Send Message
```
POST /api/chat/send
```

Sends a message to the AI therapist and receives a response.

**Request:**
```json
{
  "message": "I'm feeling really overwhelmed",
  "session_id": "session-123",
  "history": [
    {
      "role": "user",
      "content": "I've been stressed lately"
    },
    {
      "role": "assistant",
      "content": "I hear you. What's been going on?"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "response": "That sounds really tough. Let's break it down...",
  "session_id": "session-123",
  "metadata": {
    "session_type": "venting",
    "sentiment": "Negative",
    "crisis_level": "none",
    "quality_score": 82,
    "strategy_used": "empathy_first"
  },
  "resources": [
    {
      "title": "Stress Management Tips",
      "url": "https://example.com/stress",
      "description": "Evidence-based strategies for managing stress",
      "source": "trusted"
    }
  ]
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid input
- `500`: Server error

---

### Journal

#### Create Entry
```
POST /api/journal/create
```

Creates a new journal entry.

**Request:**
```json
{
  "title": "A rough week",
  "content": "This week has been emotionally exhausting...",
  "mood": "sad",
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "success": true,
  "entry_id": "entry-456",
  "created_at": "2026-04-09T10:30:00Z",
  "message": "Journal entry saved"
}
```

#### Get Entries
```
GET /api/journal/list?user_id=user-123&limit=10
```

Lists user's journal entries.

**Response:**
```json
{
  "success": true,
  "entries": [
    {
      "id": "entry-456",
      "title": "A rough week",
      "content": "This week has been emotionally exhausting...",
      "mood": "sad",
      "created_at": "2026-04-09T10:30:00Z"
    }
  ],
  "count": 1,
  "total": 1
}
```

---

### Check-In

#### Record Check-In
```
POST /api/check-in/record
```

Records a daily vibe check.

**Request:**
```json
{
  "mood": 55,
  "energy": 40,
  "stress": 65,
  "note": "Feeling overwhelmed with work",
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "success": true,
  "checkin_id": "checkin-789",
  "created_at": "2026-04-09T10:30:00Z"
}
```

**Fields:**
- `mood`: 0-100 (0=darkest, 100=brightest)
- `energy`: 0-100 (0=depleted, 100=energized)
- `stress`: 0-100 (0=calm, 100=overwhelmed)
- `note`: Optional short text

---

### Feedback

#### Submit Response Feedback
```
POST /api/feedback/submit
```

Provides feedback on an AI response.

**Request:**
```json
{
  "response_id": "response-123",
  "session_id": "session-123",
  "rating": "helpful",
  "comment": "This really helped me think through the issue",
  "quality_score": 85
}
```

**Response:**
```json
{
  "success": true,
  "message": "Feedback received, thank you!"
}
```

**Rating Options:**
- `helpful`
- `somewhat_helpful`
- `not_helpful`
- `off_topic`

---

### Insights

#### Get Analytics
```
GET /api/insights/analytics?user_id=user-123&days=7
```

Retrieves analytics and insights for user.

**Response:**
```json
{
  "success": true,
  "period": "7_days",
  "mood_trend": "improving",
  "mood_average": 62,
  "energy_average": 48,
  "stress_average": 58,
  "top_patterns": [
    {
      "pattern": "Stress increases on Monday mornings",
      "confidence": 0.85,
      "recommendation": "Plan weekend recovery time"
    }
  ],
  "insights": [
    {
      "type": "positive_trend",
      "message": "Your mood has been trending upward this week",
      "icon": "trend_up"
    }
  ]
}
```

---

### Sessions

#### Get Session List
```
GET /api/sessions?user_id=user-123&limit=10
```

Returns paginated list of user's sessions.

**Response:**
```json
{
  "success": true,
  "sessions": [
    {
      "id": "session-001",
      "title": "Feeling overwhelmed with work",
      "created_at": "2026-04-09T10:30:00Z",
      "message_count": 12,
      "quality_score": 78,
      "is_crisis": false
    }
  ],
  "count": 1,
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

#### Get Session Details
```
GET /api/sessions/[sessionId]
```

Returns detailed information about a specific session.

**Response:**
```json
{
  "success": true,
  "session": {
    "id": "session-001",
    "title": "Feeling overwhelmed with work",
    "user_id": "user-123",
    "created_at": "2026-04-09T10:30:00Z",
    "updated_at": "2026-04-09T11:00:00Z",
    "messages": [
      {
        "id": "msg-1",
        "content": "I'm feeling stressed",
        "role": "user",
        "timestamp": "2026-04-09T10:30:00Z"
      },
      {
        "id": "msg-2",
        "content": "I hear you. What's on your mind?",
        "role": "assistant",
        "timestamp": "2026-04-09T10:31:00Z",
        "quality_score": 85,
        "strategy_used": "empathy_first"
      }
    ],
    "feedback": [
      {
        "message_id": "msg-2",
        "rating": "helpful",
        "timestamp": "2026-04-09T10:32:00Z"
      }
    ],
    "quality_score": 85,
    "is_crisis": false
  }
}
```

---

### Strategies

#### Get Strategy Recommendations
```
GET /api/strategies/recommend?session_type=venting&user_id=user-123
```

Returns recommended strategies based on session type and user preferences.

**Query Parameters:**
- `session_type`: venting, problem_solving, validation_seeking, information_seeking, crisis, chitchat
- `user_id`: Optional, improves recommendations

**Response:**
```json
{
  "success": true,
  "session_type": "venting",
  "user_id": "user-123",
  "primary_strategy": "empathy_first",
  "ranked_strategies": [
    "empathy_first",
    "more_validation",
    "more_warmth"
  ],
  "strategy_details": {
    "empathy_first": {
      "name": "empathy_first",
      "description": "Lead with empathy and understanding before offering solutions",
      "rank": 1
    },
    "more_validation": {
      "name": "more_validation",
      "description": "Focus on affirming and validating the user's feelings",
      "rank": 2
    }
  },
  "user_preferences": {
    "verbosity_preference": "medium",
    "resource_preference": "moderate",
    "tone_preference": "warm",
    "satisfaction_score": 78
  }
}
```

---

## Admin-Protected Endpoints

**Authorization Required**: All admin endpoints require either:
- `Authorization: Bearer <token>` header, or
- `x-admin-token: <token>` header

Set `ADMIN_AUTH_TOKEN` environment variable with a secure token.

---

### Metrics

#### Get System Metrics
```
GET /api/metrics
```

Returns current system performance metrics.

**Response:**
```json
{
  "success": true,
  "metrics": {
    "timestamp": "2026-04-09T10:30:00Z",
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

**Metrics Explained:**
- `avg_response_quality`: 0-100 average quality score
- `avg_response_time_ms`: Average response time in milliseconds
- `total_requests`: Total number of requests processed
- `error_count`: Number of errors encountered
- `error_rate`: Percentage of requests that failed
- `crisis_detections`: Number of crisis patterns detected
- `regenerations`: Number of times responses were regenerated
- `user_satisfaction_rate`: Percentage of satisfied users

---

### Alerts

#### Get Active Alerts
```
GET /api/alerts
```

Returns list of active (unacknowledged) system alerts.

**Response:**
```json
{
  "success": true,
  "alerts": [
    {
      "id": "alert-123",
      "alert_type": "quality_decline",
      "severity": "warning",
      "message": "Response quality dropped to 45",
      "details": {
        "response_quality": 45
      },
      "created_at": "2026-04-09T10:30:00Z",
      "acknowledged": false
    },
    {
      "id": "alert-456",
      "alert_type": "crisis_escalation",
      "severity": "critical",
      "message": "Crisis detected: critical",
      "session_id": "session-789",
      "details": {
        "severity": "critical"
      },
      "created_at": "2026-04-09T10:25:00Z",
      "acknowledged": false
    }
  ],
  "count": 2
}
```

**Alert Types:**
- `crisis_escalation`: Crisis pattern detected
- `quality_decline`: Response quality below threshold
- `api_error`: External API failure
- `pattern_detected`: Notable pattern detected
- `user_support_needed`: User satisfaction declining

**Severity Levels:**
- `critical`: Immediate action required
- `warning`: Should be reviewed soon
- `info`: Informational

#### Acknowledge Alert
```
POST /api/alerts/acknowledge
```

Marks an alert as acknowledged (resolved).

**Request:**
```json
{
  "alertId": "alert-123",
  "notes": "Investigated and resolved in deployment v2.1.0"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Alert acknowledged successfully",
  "alertId": "alert-123"
}
```

---

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "message": "Error description",
  "status": 400
}
```

### Common Status Codes

| Code | Meaning | Solution |
|------|---------|----------|
| `200` | Success | No action needed |
| `201` | Created | Resource successfully created |
| `400` | Bad Request | Check request format and parameters |
| `401` | Unauthorized | Missing or invalid authentication token |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource doesn't exist |
| `429` | Rate Limited | Too many requests, wait and retry |
| `500` | Server Error | Server error, contact support |
| `503` | Service Unavailable | Service temporarily down, retry later |

---

## Rate Limiting

**Public Endpoints**: 60 requests/minute per IP
**Admin Endpoints**: 100 requests/minute per token

**Limit Headers**:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

---

## Authentication

### Admin Token Format

Generate a secure token:
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Store in environment variable:
```bash
export ADMIN_AUTH_TOKEN="your_64_char_token_here"
```

Use in requests:
```bash
curl -H "Authorization: Bearer $ADMIN_AUTH_TOKEN" \
  https://your-domain.com/api/metrics
```

Or with custom header:
```bash
curl -H "x-admin-token: $ADMIN_AUTH_TOKEN" \
  https://your-domain.com/api/metrics
```

---

## Examples

### cURL Examples

**Send Message:**
```bash
curl -X POST https://your-domain.com/api/chat/send \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I am feeling overwhelmed",
    "session_id": "session-123"
  }'
```

**Get Metrics (Admin):**
```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://your-domain.com/api/metrics
```

**Acknowledge Alert (Admin):**
```bash
curl -X POST https://your-domain.com/api/alerts/acknowledge \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "alertId": "alert-123",
    "notes": "Resolved"
  }'
```

---

## Webhook Events (Future)

Future releases will support webhook events for:
- Crisis detection
- User milestone achievements
- System alerts
- User satisfaction changes

See CHANGELOG.md for release timeline.

---

## Versioning

API follows semantic versioning. Breaking changes will be:
1. Announced in CHANGELOG.md
2. Released in major version bump
3. Supported for 2 versions with deprecation warnings

Current API version: **v1.0.0**

---

## Support

- **Documentation**: [README.md](./README.md)
- **Deployment**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Testing**: [TESTING.md](./TESTING.md)
- **Issues**: Report at GitHub Issues
