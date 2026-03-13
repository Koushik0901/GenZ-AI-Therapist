const WINDOW_MS = 60_000;
const MAX_REQUESTS = 8;

type Bucket = {
  count: number;
  resetAt: number;
};

const buckets = new Map<string, Bucket>();

function cleanup(now: number) {
  for (const [key, bucket] of buckets.entries()) {
    if (bucket.resetAt <= now) {
      buckets.delete(key);
    }
  }
}

function getClientKey(request: Request, userId?: string | null) {
  if (userId) {
    return `user:${userId}`;
  }

  const forwardedFor = request.headers.get("x-forwarded-for");
  if (forwardedFor) {
    const ip = forwardedFor.split(",")[0]?.trim();
    if (ip) {
      return `ip:${ip}`;
    }
  }

  const realIp = request.headers.get("x-real-ip");
  if (realIp) {
    return `ip:${realIp}`;
  }

  return "shared:anonymous";
}

export function limitChatSend(request: Request, userId?: string | null) {
  const now = Date.now();
  cleanup(now);

  const key = getClientKey(request, userId);
  const existing = buckets.get(key);

  if (!existing || existing.resetAt <= now) {
    buckets.set(key, {
      count: 1,
      resetAt: now + WINDOW_MS,
    });

    return {
      ok: true,
      remaining: MAX_REQUESTS - 1,
      retryAfterSeconds: 60,
    } as const;
  }

  if (existing.count >= MAX_REQUESTS) {
    return {
      ok: false,
      remaining: 0,
      retryAfterSeconds: Math.max(1, Math.ceil((existing.resetAt - now) / 1000)),
    } as const;
  }

  existing.count += 1;
  buckets.set(key, existing);

  return {
    ok: true,
    remaining: Math.max(0, MAX_REQUESTS - existing.count),
    retryAfterSeconds: Math.max(1, Math.ceil((existing.resetAt - now) / 1000)),
  } as const;
}
