/**
 * pose.js — MediaPipe Pose Landmarker wrapper
 *
 * Uses @mediapipe/tasks-vision PoseLandmarker (runs entirely in the browser).
 * Exports a thin API for the rest of the app to consume.
 */

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

// MediaPipe Pose landmark indices we care about
export const LANDMARK = {
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
};

let poseLandmarker = null;

/**
 * Initialise the PoseLandmarker. Call once at app start.
 * @returns {Promise<PoseLandmarker>}
 */
export async function initPose() {
  // Load WASM runtime from local files (no internet required)
  const vision = await FilesetResolver.forVisionTasks("/wasm");

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/model/pose_landmarker_lite.task",
      delegate: "GPU", // falls back to CPU automatically
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  return poseLandmarker;
}

/**
 * Run pose detection on a video frame.
 * @param {HTMLVideoElement} video
 * @param {number} timestampMs  monotonically increasing timestamp (e.g. performance.now())
 * @returns {{ landmarks: Array|null, worldLandmarks: Array|null }}
 */
export function detectPose(video, timestampMs) {
  if (!poseLandmarker) throw new Error("PoseLandmarker not initialised. Call initPose() first.");

  const result = poseLandmarker.detectForVideo(video, timestampMs);

  if (!result.landmarks || result.landmarks.length === 0) {
    return { landmarks: null, worldLandmarks: null };
  }

  return {
    landmarks: result.landmarks[0],       // normalised [0,1] coords
    worldLandmarks: result.worldLandmarks?.[0] ?? null, // real-world coords (metres)
  };
}

/**
 * Extract the key joint positions we need for squat analysis.
 * Coordinates are normalised [0,1]; multiply by video width/height for pixels.
 *
 * @param {Array} landmarks  — normalised landmarks from detectPose()
 * @returns {{ leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle, leftShoulder, rightShoulder }}
 */
export function extractJoints(landmarks) {
  if (!landmarks) return null;

  const get = (idx) => ({
    x: landmarks[idx].x,
    y: landmarks[idx].y,
    z: landmarks[idx].z,
    visibility: landmarks[idx].visibility,
  });

  return {
    leftHip: get(LANDMARK.LEFT_HIP),
    rightHip: get(LANDMARK.RIGHT_HIP),
    leftKnee: get(LANDMARK.LEFT_KNEE),
    rightKnee: get(LANDMARK.RIGHT_KNEE),
    leftAnkle: get(LANDMARK.LEFT_ANKLE),
    rightAnkle: get(LANDMARK.RIGHT_ANKLE),
    leftShoulder: get(LANDMARK.LEFT_SHOULDER),
    rightShoulder: get(LANDMARK.RIGHT_SHOULDER),
  };
}

/**
 * Convert normalised landmark coords → pixel coords.
 */
export function toPixel(normCoord, videoWidth, videoHeight) {
  return {
    x: Math.max(0, Math.min(Math.round(normCoord.x * videoWidth), videoWidth - 1)),
    y: Math.max(0, Math.min(Math.round(normCoord.y * videoHeight), videoHeight - 1)),
  };
}

/**
 * Create a DrawingUtils helper for drawing landmarks on a canvas.
 */
export function createDrawingUtils(ctx) {
  return new DrawingUtils(ctx);
}

export { PoseLandmarker };
