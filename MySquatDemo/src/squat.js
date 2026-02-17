/**
 * squat.js — Squat state machine
 *
 * Tracks hip/knee vertical ratio to detect squat phases and count reps.
 * Triggers depth queries at key moments (standing, bottom of squat).
 */

// ── States ──
export const STATE = {
  IDLE: "IDLE",             // waiting for subject to be detected
  STANDING: "STANDING",     // upright position
  DESCENDING: "DESCENDING", // going down
  AT_BOTTOM: "AT_BOTTOM",   // at the bottom of the squat
  ASCENDING: "ASCENDING",   // coming back up
};

// ── Thresholds (normalised Y ratio: hip.y / knee.y) ──
// When hip.y is close to knee.y the ratio approaches 1.0 → deep squat
// When standing, hip.y is much smaller than knee.y → ratio < threshold
const SQUAT_ENTER_THRESHOLD = 0.92;  // hip/knee Y ratio above this = entering squat
const SQUAT_BOTTOM_THRESHOLD = 0.96; // ratio above this = near bottom
const SQUAT_EXIT_THRESHOLD = 0.88;   // ratio below this = standing back up

const MIN_REP_DURATION_MS = 800;     // ignore reps faster than this (noise)

/**
 * Create a new SquatTracker instance.
 * @param {object} opts
 * @param {number} opts.repsPerSet — reps before triggering LLM feedback
 * @param {function} opts.onRepComplete — called with rep data when a rep finishes
 * @param {function} opts.onSetComplete — called with all rep data when repsPerSet reached
 * @param {function} opts.onStateChange — called with (newState, oldState)
 * @param {function} opts.onDepthRequest — called with (phase, joints) to trigger depth query
 */
export function createSquatTracker(opts = {}) {
  const repsPerSet = opts.repsPerSet || 5;
  const onRepComplete = opts.onRepComplete || (() => {});
  const onSetComplete = opts.onSetComplete || (() => {});
  const onStateChange = opts.onStateChange || (() => {});
  const onDepthRequest = opts.onDepthRequest || (() => {});

  let state = STATE.IDLE;
  let repCount = 0;
  let reps = [];               // completed reps in current set
  let repStartTime = null;
  let startDepth = null;       // depth at standing position (metres)
  let bottomDepth = null;      // depth at bottom of squat (metres)
  let depthRequestPending = false;

  // Smoothing: keep a rolling window of ratios to reduce jitter
  const SMOOTH_WINDOW = 5;
  let ratioHistory = [];

  function smoothedRatio(rawRatio) {
    ratioHistory.push(rawRatio);
    if (ratioHistory.length > SMOOTH_WINDOW) ratioHistory.shift();
    return ratioHistory.reduce((a, b) => a + b, 0) / ratioHistory.length;
  }

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

    const { leftHip, rightHip, leftKnee, rightKnee } = joints;

    // Average hip and knee Y (normalised, Y increases downward)
    const hipY = (leftHip.y + rightHip.y) / 2;
    const kneeY = (leftKnee.y + rightKnee.y) / 2;

    if (kneeY < 0.01) return; // avoid division by ~zero

    const rawRatio = hipY / kneeY;
    const ratio = smoothedRatio(rawRatio);

    switch (state) {
      case STATE.IDLE:
        if (ratio < SQUAT_EXIT_THRESHOLD) {
          setState(STATE.STANDING);
        }
        break;

      case STATE.STANDING:
        if (ratio >= SQUAT_ENTER_THRESHOLD) {
          // Starting to descend
          repStartTime = now;
          setState(STATE.DESCENDING);
          // Request depth at standing position
          onDepthRequest("start", joints);
        }
        break;

      case STATE.DESCENDING:
        if (ratio >= SQUAT_BOTTOM_THRESHOLD) {
          setState(STATE.AT_BOTTOM);
          // Request depth at bottom position
          onDepthRequest("bottom", joints);
        } else if (ratio < SQUAT_EXIT_THRESHOLD) {
          // Went back up without reaching bottom — false start
          setState(STATE.STANDING);
        }
        break;

      case STATE.AT_BOTTOM:
        if (ratio < SQUAT_ENTER_THRESHOLD) {
          setState(STATE.ASCENDING);
        }
        break;

      case STATE.ASCENDING:
        if (ratio < SQUAT_EXIT_THRESHOLD) {
          // Rep complete!
          const elapsed = now - (repStartTime || now);
          if (elapsed >= MIN_REP_DURATION_MS) {
            repCount++;
            const repData = {
              repNo: repCount,
              startDepth,
              bottomDepth,
              depthChange: (startDepth != null && bottomDepth != null)
                ? +(bottomDepth - startDepth).toFixed(4)
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

          // Reset for next rep
          startDepth = null;
          bottomDepth = null;
          setState(STATE.STANDING);
        }
        break;
    }
  }

  /**
   * Set the depth value received from the depth service.
   * @param {"start"|"bottom"} phase
   * @param {number} depthMetres — metric depth in metres
   */
  function setDepth(phase, depthMetres) {
    if (phase === "start") {
      startDepth = depthMetres;
    } else if (phase === "bottom") {
      bottomDepth = depthMetres;
    }
  }

  function getState() {
    return state;
  }

  function getRepCount() {
    return repCount;
  }

  function getReps() {
    return [...reps];
  }

  function getCurrentDepths() {
    return { startDepth, bottomDepth };
  }

  function reset() {
    state = STATE.IDLE;
    repCount = 0;
    reps = [];
    repStartTime = null;
    startDepth = null;
    bottomDepth = null;
    ratioHistory = [];
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
