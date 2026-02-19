/**
 * llm-client.js — OpenAI-compatible LLM client
 *
 * Handles communication with the LLM service for both reach feedback and general chat.
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

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userMessage },
  ];

  return sendChatMessage(messages, onChunk);
}

/**
 * Send a general chat message to the LLM.
 * 
 * @param {Array<{role: string, content: string}>} messages 
 * @param {function} onChunk - called with (chunk, fullText)
 * @returns {Promise<string>}
 */
export async function sendChatMessage(messages, onChunk) {
  // Qwen3-0.6B best practices (non-thinking mode)
  const body = {
    messages,
    stream: true,
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0,
    presence_penalty: 1.5,   // reduces endless repetitions
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

  return streamResponse(resp, onChunk);
}

/**
 * Stream SSE response from OpenAI-compatible API
 */
async function streamResponse(resp, onChunk) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";
  let buffer = "";
  let tokenCount = 0;
  const startTime = performance.now();

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
          tokenCount++;
          fullText += content;
          const elapsed = (performance.now() - startTime) / 1000;
          const tps = elapsed > 0 ? (tokenCount / elapsed).toFixed(1) : 0;
          if (onChunk) onChunk(content, fullText, { tokenCount, elapsed, tps: Number(tps) });
        }
      } catch {
        // skip malformed chunks
      }
    }
  }

  const durationMs = performance.now() - startTime;
  return { text: fullText, tokenCount, durationMs };
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

