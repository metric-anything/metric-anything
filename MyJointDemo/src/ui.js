/**
 * ui.js — DOM rendering and canvas overlay
 *
 * Manages all visual output: video overlay, stats panel, rep table, feedback panel.
 */

import { PoseLandmarker } from "@mediapipe/tasks-vision";
import { LANDMARK, createDrawingUtils } from "./pose.js";
import { STATE } from "./reach-joint.js";

let drawingUtils = null;

// Latch values so they stay on screen between reps
const lastRecs = { start: null, bottom: null, change: null, time: null };

// ── Colour palette ──
const COLORS = {
  accent: "#00e5ff",
  accentDim: "rgba(0, 229, 255, 0.3)",
  success: "#00e676",
  warning: "#ffab00",
  error: "#ff5252",
  textPrimary: "#ffffff",
  textSecondary: "rgba(255, 255, 255, 0.7)",
  bg: "rgba(10, 12, 20, 0.85)",
  bgPanel: "rgba(20, 25, 40, 0.75)",
};

const STATE_LABELS = {
  [STATE.IDLE]: { text: "Locating Torso…", color: COLORS.textSecondary },
  [STATE.RESET]: { text: "Ready (Sit Back)", color: COLORS.success },
  [STATE.REACHING]: { text: "Leaning Forward…", color: COLORS.accent },
  [STATE.REACHED]: { text: "Peak Lean Reached!", color: COLORS.warning },
  [STATE.RETURNING]: { text: "Returning to Start…", color: COLORS.accent },
};

/**
 * Draw the pose skeleton + knee depth labels onto the canvas.
 */
export function drawPoseOverlay(ctx, canvas, landmarks, joints, depths, state) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!landmarks) return;

  if (!drawingUtils) {
    drawingUtils = createDrawingUtils(ctx);
  }

  // Draw connectors
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "rgba(0, 229, 255, 0.4)",
    lineWidth: 2,
  });

  // Draw all landmarks as small dots
  drawingUtils.drawLandmarks(landmarks, {
    color: "rgba(255, 255, 255, 0.6)",
    lineWidth: 1,
    radius: 3,
  });

  // Highlight knees with larger markers
  if (joints) {
    const w = canvas.width;
    const h = canvas.height;

    const markers = [
      { joint: joints.leftShoulder, label: "LS", color: COLORS.success }, // Body
      { joint: joints.rightShoulder, label: "RS", color: COLORS.success },
    ];

    markers.forEach(({ joint, label, color }, i) => {
      const px = joint.x * w;
      const py = joint.y * h;

      // Glowing marker
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = color; // Use specific color
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label
      // We don't map individual depths here easily because depths are aggregated (start/bottom).
      // So just show the label for now.
      ctx.font = "10px 'Inter', monospace";
      ctx.fillStyle = "rgba(255,255,255,0.8)";
      ctx.fillText(label, px + 12, py + 4);
    });
  }

  // Visual clue border showing the current phase
  if (state) {
     const sl = STATE_LABELS[state] || STATE_LABELS[STATE.IDLE];
     ctx.strokeStyle = sl.color;
     ctx.lineWidth = 12;
     ctx.strokeRect(0, 0, canvas.width, canvas.height);
  }
}

// Cache DOM elements to avoid lookups every frame
const uiCache = {
  stateEl: null,
  repEl: null,
  startDepthEl: null,
  bottomDepthEl: null,
  changeEl: null,
  timeEl: null,
};

function getUiElements() {
  if (!uiCache.stateEl) {
    uiCache.stateEl = document.getElementById("squat-state");
    uiCache.repEl = document.getElementById("rep-count");
    uiCache.startDepthEl = document.getElementById("start-depth");
    uiCache.bottomDepthEl = document.getElementById("bottom-depth");
    uiCache.changeEl = document.getElementById("depth-change");
    uiCache.timeEl = document.getElementById("time-spent");
  }
  return uiCache;
}

/**
 * Update the stats panel in the DOM.
 */
export function updateStatsPanel(state, repCount, repsPerSet, depths) {
  const { stateEl, repEl, startDepthEl, bottomDepthEl, changeEl, timeEl } = getUiElements();

  const sl = STATE_LABELS[state] || STATE_LABELS[STATE.IDLE];
  if (stateEl) {
    stateEl.textContent = sl.text;
    stateEl.style.color = sl.color;
  }

  if (repEl) repEl.textContent = `${repCount} / ${repsPerSet}`;

  // Update latched values
  if (depths.startDepth != null) lastRecs.start = depths.startDepth;
  if (depths.bottomDepth != null) {
    lastRecs.bottom = depths.bottomDepth;
    if (lastRecs.start != null) {
      lastRecs.change = lastRecs.start - lastRecs.bottom;
    }
  }
  if (depths.timeSpent != null) lastRecs.time = depths.timeSpent;

  // Clear old bottom/time metrics when a new reach actually begins
  if (state === STATE.REACHING && depths.bottomDepth == null) {
    lastRecs.bottom = null;
    lastRecs.change = null;
    lastRecs.time = null;
  }
  
  // If we reset entirely (e.g. tracking lost)
  if (state === STATE.IDLE) {
    lastRecs.start = null;
    lastRecs.bottom = null;
    lastRecs.change = null;
    lastRecs.time = null;
  }

  if (startDepthEl) {
    startDepthEl.textContent = lastRecs.start != null
      ? `${lastRecs.start.toFixed(3)}m` : "—";
  }
  
  if (bottomDepthEl) {
    bottomDepthEl.textContent = lastRecs.bottom != null
      ? `${lastRecs.bottom.toFixed(3)}m` : "—";
  }
  
  if (changeEl) {
    if (lastRecs.change != null) {
      changeEl.textContent = `${lastRecs.change > 0 ? "+" : ""}${lastRecs.change.toFixed(3)}m`;
      changeEl.style.color = COLORS.accent;
    } else {
      changeEl.textContent = "—";
      changeEl.style.color = COLORS.textSecondary;
    }
  }

  if (timeEl) {
     timeEl.textContent = lastRecs.time != null ? `${lastRecs.time.toFixed(2)}s` : "—";
  }
}

/**
 * Add a completed rep to the history table.
 */
export function addRepToTable(rep) {
  const tbody = document.getElementById("rep-table-body");
  if (!tbody) return;

  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${rep.repNo}</td>
    <td>${rep.startDepth?.toFixed(3) ?? "—"}</td>
    <td>${rep.bottomDepth?.toFixed(3) ?? "—"}</td>
    <td class="${rep.depthChange != null && rep.depthChange < 0 ? 'val-good' : 'val-warn'}">${rep.depthChange?.toFixed(3) ?? "—"}</td>
    <td>${rep.timeSpent}s</td>
  `;
  row.classList.add("rep-row-enter");
  tbody.appendChild(row);

  // Scroll to bottom
  const tableContainer = tbody.closest(".rep-history");
  if (tableContainer) tableContainer.scrollTop = tableContainer.scrollHeight;
}

/**
 * Clear the rep history table.
 */
export function clearRepTable() {
  const tbody = document.getElementById("rep-table-body");
  if (tbody) tbody.innerHTML = "";
}

/**
 * Show/update the LLM feedback panel.
 */
export function showFeedback(text, isStreaming) {
  const panel = document.getElementById("feedback-panel");
  const content = document.getElementById("feedback-content");
  const spinner = document.getElementById("feedback-spinner");

  if (panel) panel.classList.add("visible");
  if (content) content.textContent = text;
  if (spinner) spinner.style.display = isStreaming ? "inline-block" : "none";
}

/**
 * Hide the feedback panel.
 */
export function hideFeedback() {
  const panel = document.getElementById("feedback-panel");
  if (panel) panel.classList.remove("visible");
}

/**
 * Update the service status indicators.
 */
export function updateServiceStatus(depthResult, llmResult) {
  const depthEl = document.getElementById("depth-status");
  const llmEl = document.getElementById("llm-status");

  // depthResult / llmResult can be strings (old api) or objects (new api)
  const depthStatus = typeof depthResult === "string" ? depthResult : depthResult?.status || "error";
  const llmStatus = typeof llmResult === "string" ? llmResult : llmResult?.status || "error";
  
  const depthModel = depthResult?.active_model || "";
  const llmModel = llmResult?.active_model || "";

  if (depthEl) {
    depthEl.className = `status-dot ${depthStatus === "ok" ? "status-ok" : "status-err"}`;
    depthEl.title = `Depth: ${depthStatus}`;
    
    const labelSpan = depthEl.nextElementSibling;
    if (labelSpan) {
      labelSpan.textContent = depthModel ? `Depth (${depthModel})` : "Depth";
    }
  }
  if (llmEl) {
    llmEl.className = `status-dot ${llmStatus === "ok" ? "status-ok" : "status-err"}`;
    llmEl.title = `LLM: ${llmStatus}`;
    
    const labelSpan = llmEl.nextElementSibling;
    if (labelSpan) {
      labelSpan.textContent = llmModel ? `LLM (${llmModel})` : "LLM";
    }
  }
}

/**
 * Update the FPS counter.
 */
export function updateFPS(fps) {
  const el = document.getElementById("fps");
  if (el) el.textContent = `${fps} FPS`;
}
