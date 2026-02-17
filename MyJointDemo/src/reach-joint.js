/**
 * reach-joint.js — Reach Gesture Debug tracker
 *
 * Tracks a "Reach" gesture for debugging:
 * 1. Start: "W" pose (sitting, elbows down, hands up).
 * 2. Action: Reach forward with one hand (Z-axis depth change).
 * 3. Return: Return to "W" pose.
 */

// ── States ──
export const STATE = {
  IDLE: "IDLE",             // waiting for subject / W-pose
  RESET: "RESET",           // "W" pose detected (Ready)
  REACHING: "REACHING",     // Hand moving forward
  REACHED: "REACHED",       // Peak reach (trigger depth)
  RETURNING: "RETURNING",   // Hand moving back
};

// ── Thresholds ──
// "W" Pose: Wrist Y must be < Elbow Y (higher on screen)
// Reach: Wrist Z must be significantly less than Shoulder Z (closer to camera)
const REACH_THRESHOLD = -0.15; // Wrist Z is this much closer than Shoulder Z
const RETURN_THRESHOLD = -0.05; // Wrist Z is moving back to Shoulder Z
const MIN_REP_DURATION_MS = 500;

/**
 * Create a new ReachTracker instance.
 * @param {object} opts
 */
export function createSquatTracker(opts = {}) {
  const repsPerSet = opts.repsPerSet || 5;
  const onRepComplete = opts.onRepComplete || (() => {});
  const onSetComplete = opts.onSetComplete || (() => {});
  const onStateChange = opts.onStateChange || (() => {});
  const onDepthRequest = opts.onDepthRequest || (() => {});

  let state = STATE.IDLE;
  let repCount = 0;
  let reps = [];
  let repStartTime = null;
  let startDepth = null;
  let bottomDepth = null; // reused "bottom" for "reach" depth

  function setState(newState) {
    if (newState !== state) {
      const old = state;
      state = newState;
      onStateChange(newState, old);
    }
  }

  /**
   * Feed new joint data each frame.
   * @param {object} joints — from extractJoints()
   * @param {number} now — timestamp in ms
   */
  function update(joints, now) {
    if (!joints) {
      setState(STATE.IDLE);
      return;
    }

    const {
      leftShoulder, rightShoulder,
      leftElbow, rightElbow,
      leftWrist, rightWrist
    } = joints;

    // Helper: Check if a side is in "W" pose (Wrist above Elbow)
    // Note: Y increases downward in screen coords (0=top)
    // So Wrist.y < Elbow.y means Wrist is higher visually (like hands up)
    const leftIsUp = leftWrist.y < leftElbow.y;
    const rightIsUp = rightWrist.y < rightElbow.y;

    // Helper: Calculate Z-depth relative to shoulder (negative = closer to camera)
    // MediaPipe World Landmarks: Z is meters, origin at hip center usually.
    // Use relative Z to shoulder to be robust to body position.
    const leftReachZ = leftWrist.z - leftShoulder.z;
    const rightReachZ = rightWrist.z - rightShoulder.z;

    // Active hand: usually the one reaching further (min Z)
    const activeReachZ = Math.min(leftReachZ, rightReachZ);

    switch (state) {
      case STATE.IDLE:
      case STATE.RESET: // "W" Pose
        // Ideally both hands up, but at least one for flexibility? Let's say both for "W".
        if (leftIsUp && rightIsUp) {
            if (state !== STATE.RESET) setState(STATE.RESET);
            
            // Check for Reach start
            if (activeReachZ < REACH_THRESHOLD) {
                repStartTime = now;
                setState(STATE.REACHING);
                onDepthRequest("start", joints); // Request "start" depth (torso/shoulder)
            }
        } else {
            setState(STATE.IDLE);
        }
        break;

      case STATE.REACHING:
        // We are reaching out. Wait for peak or just transition to REACHED?
        // Let's transition to REACHED immediately when threshold passed, 
        // effectively treating "Reaching" as "Descending".
        // Actually, let's just trigger REACHED when deep enough.
        if (activeReachZ < REACH_THRESHOLD) {
             setState(STATE.REACHED);
             // Request "reach" depth (hand) immediately upon reaching threshold
             console.log("Peak reach detected, requesting depth...", { activeReachZ });
             onDepthRequest("bottom", joints); 
        } else if (activeReachZ > RETURN_THRESHOLD) {
            // Aborted reach
            setState(STATE.RESET);
        }
        break;

      case STATE.REACHED:
        // User is holding reach or returning.
        if (activeReachZ > RETURN_THRESHOLD) {
            setState(STATE.RETURNING);
        }
        break;

      case STATE.RETURNING:
        // Check if back to W pose
        if ((leftIsUp && rightIsUp) && activeReachZ > RETURN_THRESHOLD) {
           // Rep complete
           const elapsed = now - (repStartTime || now);
           if (elapsed >= MIN_REP_DURATION_MS) {
                repCount++;
                const repData = {
                  repNo: repCount,
                  startDepth, // Body depth
                  bottomDepth, // Hand depth
                  depthChange: (startDepth != null && bottomDepth != null)
                    ? +(startDepth - bottomDepth).toFixed(4) // Body - Hand (positive delta)
                    : null,
                  timeSpent: +(elapsed / 1000).toFixed(2),
                  timestamp: now,
                };
                reps.push(repData);
                onRepComplete(repData);

                if (reps.length >= repsPerSet) {
                  onSetComplete([...reps]);
                  reps = [];
                  repCount = 0;
                }
           }
           
           // Reset
           startDepth = null;
           bottomDepth = null;
           setState(STATE.RESET);
        }
        break;
    }
  }

  function setDepth(phase, depthMetres) {
    console.log(`setDepth: ${phase} = ${depthMetres}`);
    if (phase === "start") {
      startDepth = depthMetres;
    } else if (phase === "bottom") {
      bottomDepth = depthMetres;
    }
  }

  function getState() { return state; }
  function getRepCount() { return repCount; }
  function getReps() { return [...reps]; }
  function getCurrentDepths() { return { startDepth, bottomDepth }; }
  function reset() {
    state = STATE.IDLE;
    repCount = 0;
    reps = [];
    repStartTime = null;
    startDepth = null;
    bottomDepth = null;
  }

  return {
    update,
    setDepth,
    getState,
    getRepCount,
    getReps,
    getCurrentDepths,
    reset,
  };
}
