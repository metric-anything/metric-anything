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
  IDLE: "IDLE",             // building baseline
  RESET: "RESET",           // Ready (Sitting Back)
  REACHING: "REACHING",     // Leaning forward
  REACHED: "REACHED",       // Peak lean
  RETURNING: "RETURNING",   // Returning
};

// ── Thresholds ──
// Lean: Torso Z becomes more negative as you lean toward the camera 
// (relative to hips root in MediaPipe).
const LEAN_START_THRESHOLD = -0.04; 
const LEAN_PEAK_THRESHOLD = -0.25;  // Requires a deeper forward lean
const RETURN_THRESHOLD = -0.10;     // Must return past this point to finish rep
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
  let baselineZ = null;
  let idleFrames = 0;

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

    const { leftShoulder, rightShoulder } = joints;
    const currentZ = (leftShoulder.z + rightShoulder.z) / 2;

    if (baselineZ === null) {
      baselineZ = currentZ;
    }
    const deltaZ = currentZ - baselineZ;

    switch (state) {
      case STATE.IDLE:
        // Update baseline slowly (leaky integrator)
        baselineZ = 0.9 * baselineZ + 0.1 * currentZ;
        if (Math.abs(deltaZ) < 0.05) {
           idleFrames++;
           if (idleFrames > 10) setState(STATE.RESET);
        } else {
           idleFrames = 0;
        }
        break;

      case STATE.RESET:
        // Keep gently updating baseline while sitting back
        baselineZ = 0.95 * baselineZ + 0.05 * currentZ;
        
        if (deltaZ < LEAN_START_THRESHOLD) {
            repStartTime = now;
            setState(STATE.REACHING);
            onDepthRequest("start", joints); // Request "start" depth (Torso)
        }
        break;

      case STATE.REACHING:
        if (deltaZ < LEAN_PEAK_THRESHOLD) {
             setState(STATE.REACHED);
             console.log("Peak lean detected, requesting depth...", { deltaZ });
             onDepthRequest("bottom", joints); // Peak depth
        } else if (deltaZ > (LEAN_START_THRESHOLD / 2)) {
            // Aborted lean
            setState(STATE.RESET);
        }
        break;

      case STATE.REACHED:
        if (deltaZ > RETURN_THRESHOLD) {
            setState(STATE.RETURNING);
        }
        break;

      case STATE.RETURNING:
        // Check if returned AND we've securely received the bottom depth from the network
        if (deltaZ > RETURN_THRESHOLD && bottomDepth !== null) {
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
  function getCurrentDepths() { 
     let timeSpent = null;
     if (state === STATE.REACHING || state === STATE.REACHED || state === STATE.RETURNING) {
        if (repStartTime) {
           timeSpent = (performance.now() - repStartTime) / 1000;
        }
     }
     return { startDepth, bottomDepth, timeSpent }; 
  }
  function reset() {
    state = STATE.IDLE;
    repCount = 0;
    reps = [];
    repStartTime = null;
    startDepth = null;
    bottomDepth = null;
    baselineZ = null;
    idleFrames = 0;
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
