/**
 * depth-client.js — Thin client for the Depth Estimation API
 *
 * Captures a video frame, encodes it as base64 JPEG, and sends it to
 * the depth service to get metric depth at specific pixel coordinates.
 */

// In production, requests are proxied through nginx (same origin, no CORS).
// For local dev, set VITE_DEPTH_URL=http://localhost:8081
const DEPTH_URL = import.meta.env.VITE_DEPTH_URL || "/api/depth";
const DEPTH_MODEL = import.meta.env.VITE_DEPTH_MODEL || "dav2-small";

// Reusable off-screen canvas for frame capture
let captureCanvas = null;
let captureCtx = null;

/**
 * Capture the current video frame as a base64 JPEG string (no data: prefix).
 * @param {HTMLVideoElement} video
 * @param {object} [options]
 * @param {number} [options.quality=0.7]  JPEG quality 0–1
 * @param {number} [options.width=0]      Target width (0 = original size)
 * @returns {string} base64-encoded JPEG
 */
export function captureFrameBase64(video, { quality = 0.7, width = 0 } = {}) {
  if (!captureCanvas) {
    captureCanvas = document.createElement("canvas");
    captureCtx = captureCanvas.getContext("2d");
  }

  let targetWidth = video.videoWidth;
  let targetHeight = video.videoHeight;

  // Downscale if requested
  if (width > 0 && width < targetWidth) {
    const scale = width / targetWidth;
    targetWidth = width;
    targetHeight = Math.round(targetHeight * scale);
  }

  // Resize canvas only if dimensions change (avoids allocation churn)
  if (captureCanvas.width !== targetWidth || captureCanvas.height !== targetHeight) {
    captureCanvas.width = targetWidth;
    captureCanvas.height = targetHeight;
  }

  captureCtx.drawImage(video, 0, 0, targetWidth, targetHeight);

  // toDataURL returns "data:image/jpeg;base64,..." — strip the prefix
  const dataUrl = captureCanvas.toDataURL("image/jpeg", quality);
  return dataUrl.split(",")[1];
}

/**
 * Query the depth service for metric depth at given pixel coordinates.
 *
 * @param {string} imageBase64  base64 JPEG (no prefix)
 * @param {Array<[number,number]>} points  pixel coords [[x,y], ...]
 * @param {string} [model]  override model name
 * @returns {Promise<{ depths: number[], model: string, inference_ms: number }>}
 */
export async function queryDepthAtPoints(imageBase64, points, model) {
  const body = {
    image_base64: imageBase64,
    points,
    model: model || DEPTH_MODEL,
  };

  const resp = await fetch(`${DEPTH_URL}/depth/points`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`Depth service error ${resp.status}: ${text}`);
  }

  return resp.json();
}

/**
 * Check if the depth service is healthy.
 * @returns {Promise<{ status: string, active_model: string }>}
 */
export async function checkDepthHealth() {
  try {
    const resp = await fetch(`${DEPTH_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (!resp.ok) return { status: "error" };
    return resp.json();
  } catch {
    return { status: "unreachable" };
  }
}
