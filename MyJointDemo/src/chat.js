import { sendChatMessage } from "./llm-client.js";

const chatContainer = document.getElementById("chat-container");
const chatForm = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const statusIndicator = document.getElementById("status-indicator");

// Store conversation history
let messages = [];

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = content;
  chatContainer.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatStats(tokenCount, durationSec, tps) {
  return `${tokenCount} tokens · ${durationSec.toFixed(1)}s · ${tps} tok/s`;
}

async function handleSubmit(e) {
  e.preventDefault();
  const text = promptInput.value.trim();
  if (!text) return;

  // Add user message to UI
  addMessage("user", text);
  messages.push({ role: "user", content: text });
  
  promptInput.value = "";
  promptInput.disabled = true;
  sendBtn.disabled = true;
  statusIndicator.textContent = "Waiting for response...";
  statusIndicator.className = "status streaming";

  // Prepare assistant message bubble
  const assistantDiv = addMessage("assistant", "");
  let fullResponse = "";
  let firstTokenTime = null;
  const startTime = performance.now();

  try {
    const result = await sendChatMessage(messages, (chunk, currentFullText, stats) => {
      if (!firstTokenTime) {
        firstTokenTime = performance.now();
        const latency = (firstTokenTime - startTime).toFixed(0);
        statusIndicator.textContent = `First token: ${latency}ms · streaming...`;
      }
      fullResponse = currentFullText;
      assistantDiv.textContent = fullResponse;

      // Live tokens/sec in status bar
      if (stats) {
        statusIndicator.textContent = `⚡ ${stats.tps} tok/s · ${stats.tokenCount} tokens`;
      }
      scrollToBottom();
    });

    // Finalize
    messages.push({ role: "assistant", content: result.text });
    const durationSec = result.durationMs / 1000;
    const finalTps = durationSec > 0 ? (result.tokenCount / durationSec).toFixed(1) : 0;
    
    // Final stats in status bar
    statusIndicator.textContent = `✓ ${formatStats(result.tokenCount, durationSec, finalTps)}`;
    statusIndicator.className = "status done";

    // Add stats footer to the message
    const statsSpan = document.createElement("span");
    statsSpan.className = "msg-stats";
    statsSpan.textContent = formatStats(result.tokenCount, durationSec, finalTps);
    assistantDiv.appendChild(statsSpan);

  } catch (err) {
    assistantDiv.textContent += `\n[Error: ${err.message}]`;
    statusIndicator.textContent = "⚠ Error";
    statusIndicator.className = "status error";
  } finally {
    promptInput.disabled = false;
    sendBtn.disabled = false;
    promptInput.focus();
  }
}

chatForm.addEventListener("submit", handleSubmit);

// General-purpose system prompt for the chat page
messages.push({ 
  role: "system", 
  content: "You are a helpful, knowledgeable assistant. Answer clearly and concisely." 
});
