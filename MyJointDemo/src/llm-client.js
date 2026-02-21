/**
 * llm-client.js — OpenAI-compatible LLM client
 *
 * Handles communication with the LLM service for both reach feedback and general chat.
 */

// Requests are proxied through Vite in dev and Nginx in prod (same origin, no CORS).
const LLM_URL = "/api/llm";

const SYSTEM_PROMPT = `你是一位专业的物理治疗师和人体运动科学家，擅长人体各种动作的生物力学分析。

你将收到包含以下内容的动作数据：
- 动作序号 (Rep number)
- 动作起始深度（距摄像头的米数）
- 动作峰值深度（距摄像头的米数）—— 动作到达的最远点
- 动作距离（米）—— 动作移动的总距离
- 每轮动作耗时（秒）

对数据进行分析，并给出以下几方面的反馈，适当分段，不要冗长：
1. 一句话对动作执行情况进行总结，包含幅度、节奏等
2. 一条具体的改进建议

保持语句精炼，鼓励为主，诚实为辅，必须使用中文。`;

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
    `peak_body=${r.bottomDepth?.toFixed(3) ?? "N/A"}m, ` +
    `lean_dist=${r.depthChange?.toFixed(3) ?? "N/A"}m, ` +
    `time=${r.timeSpent}s`
  ).join("\n");

  const userMessage = `Here is my torso lean gesture data for ${reps.length} reps:\n\n${userData}\n\nPlease analyze my lean performance.`;

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
  // Official defaults for Qwen 2.5 1.5B-Instruct
  const currentModel = window.APP_CONFIG?.LLM_MODEL || "qwen2.5:1.5b";
  
  const body = {
    model: currentModel,
    messages,
    stream: true,
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0,
    repetition_penalty: 1.1, // Official config: "repetition_penalty": 1.1
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
    const resp = await fetch(`${LLM_URL}/v1/models`, { signal: AbortSignal.timeout(3000) });
    if (!resp.ok) return { status: "error" };
    return { status: "ok" };
  } catch {
    return { status: "unreachable" };
  }
}

