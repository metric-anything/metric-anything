/**
 * ui.js — DOM rendering and canvas overlay
 *
 * Manages all visual output: video overlay, stats panel, rep table, feedback panel.
 */

import { PoseLandmarker } from "@mediapipe/tasks-vision";
import { LANDMARK, createDrawingUtils } from "./pose.js";
import { STATE } from "./squat.js";

let drawingUtils = null;

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
  [STATE.IDLE]: { text: "Waiting for pose…", color: COLORS.textSecondary },
  [STATE.STANDING]: { text: "Standing — ready", color: COLORS.success },
  [STATE.DESCENDING]: { text: "Going down ⬇", color: COLORS.accent },
  [STATE.AT_BOTTOM]: { text: "At bottom ⬇⬇", color: COLORS.warning },
  [STATE.ASCENDING]: { text: "Coming up ⬆", color: COLORS.accent },
};

/**
 * Draw the pose skeleton + knee depth labels onto the canvas.
 */
export function drawPoseOverlay(ctx, canvas, landmarks, joints, depths) {
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

    const knees = [
      { joint: joints.leftKnee, label: "L" },
      { joint: joints.rightKnee, label: "R" },
    ];

    knees.forEach(({ joint, label }, i) => {
      const px = joint.x * w;
      const py = joint.y * h;

      // Glowing knee marker
      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.accentDim;
      ctx.fill();
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Depth label (if available)
      const depthVal = depths?.[i];
      if (depthVal != null) {
        const text = `${depthVal.toFixed(3)}m`;
        ctx.font = "bold 14px 'Inter', monospace";
        ctx.fillStyle = COLORS.accent;
        ctx.textAlign = "left";
        ctx.fillText(text, px + 16, py + 5);
      }
    });
  }
}

/**
 * Update the stats panel in the DOM.
 */
export function updateStatsPanel(state, repCount, repsPerSet, depths, latency) {
  const stateEl = document.getElementById("squat-state");
  const repEl = document.getElementById("rep-count");
  const startDepthEl = document.getElementById("start-depth");
  const bottomDepthEl = document.getElementById("bottom-depth");
  const changeEl = document.getElementById("depth-change");
  const latencyEl = document.getElementById("latency");

  const sl = STATE_LABELS[state] || STATE_LABELS[STATE.IDLE];
  if (stateEl) {
    stateEl.textContent = sl.text;
    stateEl.style.color = sl.color;
  }

  if (repEl) repEl.textContent = `${repCount} / ${repsPerSet}`;

  if (startDepthEl) {
    startDepthEl.textContent = depths.startDepth != null
      ? `${depths.startDepth.toFixed(3)}m` : "—";
  }
  if (bottomDepthEl) {
    bottomDepthEl.textContent = depths.bottomDepth != null
      ? `${depths.bottomDepth.toFixed(3)}m` : "—";
  }
  if (changeEl) {
    if (depths.startDepth != null && depths.bottomDepth != null) {
      const change = depths.bottomDepth - depths.startDepth;
      changeEl.textContent = `${change >= 0 ? "+" : ""}${change.toFixed(3)}m`;
      changeEl.style.color = change < 0 ? COLORS.success : COLORS.warning;
    } else {
      changeEl.textContent = "—";
      changeEl.style.color = COLORS.textSecondary;
    }
  }

  if (latencyEl) {
    latencyEl.textContent = latency != null ? `${latency.toFixed(0)}ms` : "—";
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
export function updateServiceStatus(depthStatus, llmStatus) {
  const depthEl = document.getElementById("depth-status");
  const llmEl = document.getElementById("llm-status");

  if (depthEl) {
    depthEl.className = `status-dot ${depthStatus === "ok" ? "status-ok" : "status-err"}`;
    depthEl.title = `Depth: ${depthStatus}`;
  }
  if (llmEl) {
    llmEl.className = `status-dot ${llmStatus === "ok" ? "status-ok" : "status-err"}`;
    llmEl.title = `LLM: ${llmStatus}`;
  }
}

/**
 * Update the FPS counter.
 */
export function updateFPS(fps) {
  const el = document.getElementById("fps");
  if (el) el.textContent = `${fps} FPS`;
}
