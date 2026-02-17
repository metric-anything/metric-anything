/**
 * llm-client.js — OpenAI-compatible LLM client for coaching feedback
 *
 * Sends squat movement data to the LLM service and streams the response.
 */

// In production, requests are proxied through nginx (same origin, no CORS).
// For local dev, set VITE_LLM_URL=http://localhost:11434
const LLM_URL = import.meta.env.VITE_LLM_URL || "/api/llm";

const SYSTEM_PROMPT = `You are an expert physical therapist and human movement scientist. You specialize in biomechanics analysis for upper body reach gestures.

You will receive reach rep data that includes:
- Rep number
- Starting body depth (metres from camera)
- Peak reach depth (metres from camera) — the hand position at furthest point
- Reach distance (metres) — how far the hand travelled forward
- Time spent per rep (seconds)

Analyze the data and provide:
1. A brief assessment of reach consistency
2. Whether the extension is adequate
3. Any tempo concerns (too fast/slow)
4. One specific, actionable tip to improve

Keep your response concise (3-5 sentences). Be encouraging but honest. Must in Chinese.`;

/**
 * Get coaching feedback from the LLM.
 *
 * @param {Array<object>} reps — array of rep data objects
 * @param {function} onChunk — called with each text chunk as it streams
 * @returns {Promise<string>} — the full response text
 */
export async function getReachFeedback(reps, onChunk) {
  const userData = reps.map((r) =>
    `Rep ${r.repNo}: start_body=${r.startDepth?.toFixed(3) ?? "N/A"}m, ` +
    `peak_hand=${r.bottomDepth?.toFixed(3) ?? "N/A"}m, ` +
    `reach_dist=${r.depthChange?.toFixed(3) ?? "N/A"}m, ` +
    `time=${r.timeSpent}s`
  ).join("\n");

  const userMessage = `Here is my reach gesture data for ${reps.length} reps:\n\n${userData}\n\nPlease analyze my reach performance.`;

  const body = {
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userMessage },
    ],
    stream: true,
    temperature: 0.7,
    max_tokens: 300,
  };

  const resp = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`LLM service error ${resp.status}: ${text}`);
  }

  // Stream SSE response
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete SSE lines
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete line in buffer

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;

      try {
        const parsed = JSON.parse(data);
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) {
          fullText += content;
          if (onChunk) onChunk(content, fullText);
        }
      } catch {
        // skip malformed chunks
      }
    }
  }

  return fullText;
}

/**
 * Check if the LLM service is reachable.
 * @returns {Promise<{ status: string }>}
 */
export async function checkLLMHealth() {
  try {
    const resp = await fetch(`${LLM_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (!resp.ok) return { status: "error" };
    return { status: "ok" };
  } catch {
    return { status: "unreachable" };
  }
}
