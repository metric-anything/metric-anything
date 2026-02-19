/**
 * main.js — MySquatDemo entry point
 *
 * Orchestrates webcam capture, pose detection, squat tracking,
 * depth queries, and LLM feedback into a cohesive real-time UI.
 *
 * OPTIMIZATIONS:
 * - Camera: 720p (sufficient for pose detection, faster than 1080p)
 * - Depth:  Images downscaled to 640px before upload (~80% bandwidth saving)
 */

import "./style.css";
import { initPose, detectPose, extractJoints, toPixel } from "./pose.js";
import { createSquatTracker } from "./reach-joint.js";
import { captureFrameBase64, queryDepthAtPoints, checkDepthHealth } from "./depth-client.js";
import { getReachFeedback, checkLLMHealth } from "./llm-client.js";
import {
  drawPoseOverlay,
  updateStatsPanel,
  addRepToTable,
  clearRepTable,
  showFeedback,
  updateServiceStatus,
  updateFPS,
} from "./ui.js";

// ── Configuration ──
const REPS_PER_SET = parseInt(import.meta.env.VITE_REPS_PER_SET || "5", 10);

// ── DOM elements ──
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const loadingScreen = document.getElementById("loading-screen");
const loadingStatus = document.getElementById("loading-status");

// ── State ──
let lastFrameTime = 0;
let fpsFrames = 0;
let fpsLastUpdate = 0;
let depthInFlight = false;  // prevent concurrent depth queries

// ── Squat Tracker ──
const tracker = createSquatTracker({
  repsPerSet: REPS_PER_SET,

  onStateChange(newState, oldState) {
    // State badge is updated in the render loop
  },

  onDepthRequest(phase, joints) {
    // Query depth service (non-blocking)
    // Query depth service (non-blocking)
    // Note: We authorize concurrent requests for different phases to avoid missing data.
    // The service should handle it. If bandwidth is issue, we can debounce.

    const w = video.videoWidth;
    const h = video.videoHeight;

    // Optimization: Downscale image for depth service
    // Most depth models work at ~518px internal resolution.
    // 640px is a safe upper bound that saves huge bandwidth vs 1080p.
    const DEPTH_TARGET_WIDTH = 640;
    
    // Calculate scale factor if we need to resize
    const scale = (w > DEPTH_TARGET_WIDTH) ? (DEPTH_TARGET_WIDTH / w) : 1.0;

    // Capture scaled frame using our optimized client
    const frame = captureFrameBase64(video, { width: DEPTH_TARGET_WIDTH });

    // Dynamic targets based on phase
    // User Stat: "delta is the difference between start and end for the hand."
    // So we track Wrists for BOTH phases now.
    const targets = [
        toPixel(joints.leftWrist, w, h),
        toPixel(joints.rightWrist, w, h)
    ];

    // IMPORTANT: We must scale the query points to match the resized image!
    queryDepthAtPoints(frame, [
      [Math.round(targets[0].x * scale), Math.round(targets[0].y * scale)],
      [Math.round(targets[1].x * scale), Math.round(targets[1].y * scale)],
    ])
      .then((result) => {
        // Always take the minimum depth (closest hand)
        // This ensures we track the "active" reaching hand
        const depthVal = Math.min(result.depths[0], result.depths[1]);
        tracker.setDepth(phase, depthVal);

        // Update latency display
        const latencyEl = document.getElementById("latency");
        if (latencyEl) latencyEl.textContent = `${result.inference_ms.toFixed(0)}ms`;
      })
      .catch((err) => {
        console.warn(`Depth query failed (${phase}):`, err.message);
      })
      .finally(() => {
         // depthInFlight = false; // concurrency check removed
      });
  },

  onRepComplete(rep) {
    console.log("Rep complete:", rep);
    // Start of a new set: clear the old table
    if (rep.repNo === 1) {
      clearRepTable();
    }
    addRepToTable(rep);
  },

  onSetComplete(reps) {
    console.log("Set complete! Requesting LLM feedback…", reps);
    requestLLMFeedback(reps);
  },
});

// ── LLM Feedback ──
async function requestLLMFeedback(reps) {
  showFeedback("Analyzing your squat form…", true);

  try {
    await getReachFeedback(reps, (chunk, fullText) => {
      showFeedback(fullText, true);
    });

    // Streaming done — update UI
    const contentEl = document.getElementById("feedback-content");
    if (contentEl) {
      showFeedback(contentEl.textContent, false);
    }
  } catch (err) {
    showFeedback(`⚠ Could not get feedback: ${err.message}`, false);
  }

  // Feedback stays visible until next set replaces it.
}

// ── Health Checks ──
async function checkServices() {
  const [depthResult, llmResult] = await Promise.all([
    checkDepthHealth(),
    checkLLMHealth(),
  ]);
  updateServiceStatus(depthResult.status, llmResult.status);
}

// ── Main Loop ──
function renderLoop(timestamp) {
  requestAnimationFrame(renderLoop);

  if (!video.videoWidth || !video.videoHeight) return;

  // Sync canvas size with video
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  // Throttle pose detection to ~30 fps
  if (timestamp - lastFrameTime < 33) return;
  lastFrameTime = timestamp;

  // FPS counter
  fpsFrames++;
  if (timestamp - fpsLastUpdate >= 1000) {
    updateFPS(fpsFrames);
    fpsFrames = 0;
    fpsLastUpdate = timestamp;
  }

  // Pose detection
  const poseResult = detectPose(video, timestamp);
  const joints = extractJoints(poseResult.landmarks);

  // Update squat tracker
  tracker.update(joints, timestamp);

  // Get current depth info for overlay
  const depths = tracker.getCurrentDepths();
  const kneeDepths = depths.startDepth != null || depths.bottomDepth != null
    ? [depths.startDepth, depths.bottomDepth]
    : null;

  // Draw overlay
  drawPoseOverlay(ctx, canvas, poseResult.landmarks, joints, kneeDepths);

  // Update stats panel
  updateStatsPanel(
    tracker.getState(),
    tracker.getRepCount(),
    REPS_PER_SET,
    depths,
    null, // latency updated separately by depth callback
  );

  // Update state badge
  const stateBadge = document.getElementById("squat-state");
  if (stateBadge) {
    // Already handled by updateStatsPanel
  }
}

// ── Initialisation ──
async function init() {
  try {
    updateLoadingStatus("Requesting camera access…");
    const stream = await navigator.mediaDevices.getUserMedia({
      // Optimization: 720p is approx 2x faster to process than 1080p
      // and still offers plenty of detail for body tracking.
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    updateLoadingStatus("Loading pose model…");
    await initPose();

    updateLoadingStatus("Checking backend services…");
    await checkServices();

    // Periodic health checks
    setInterval(checkServices, 10000);

    // Hide loading screen
    if (loadingScreen) loadingScreen.classList.add("hidden");

    // Start render loop
    requestAnimationFrame(renderLoop);
    console.log("MySquatDemo initialised ✓");
  } catch (err) {
    updateLoadingStatus(`Error: ${err.message}`);
    console.error("Init failed:", err);
  }
}

function updateLoadingStatus(text) {
  if (loadingStatus) loadingStatus.textContent = text;
  console.log(`[init] ${text}`);
}

init();
