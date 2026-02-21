/**
 * depth-client.js — Thin client for the Depth Estimation API
 *
 * Captures a video frame, encodes it as base64 JPEG, and sends it to
 * the depth service to get metric depth at specific pixel coordinates.
 */

// Requests are proxied through Vite in dev and Nginx in prod (same origin, no CORS).
const DEPTH_URL = "/api/depth";
const DEPTH_MODEL = import.meta.env.VITE_DEPTH_MODEL || "dav2-small";

// Reusable off-screen canvas for frame capture
let captureCanvas = null;
let captureCtx = null;

// Reusable connection for streaming
let wsConnection = null;
let pendingResolvers = [];
let pendingRejecters = [];

const DEPTH_TARGET_WIDTH = 640;
const DEPTH_TARGET_HEIGHT = 360;

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
 * Ensures the WebSocket connection is open.
 */
function getWebSocket(model) {
  return new Promise((resolve, reject) => {
    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
      resolve(wsConnection);
      return;
    }

    if (wsConnection && wsConnection.readyState === WebSocket.CONNECTING) {
      // Wait for it to open
      const onOpen = () => {
        wsConnection.removeEventListener('open', onOpen);
        resolve(wsConnection);
      };
      wsConnection.addEventListener('open', onOpen);
      return;
    }

    // Determine WS URL
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let wsUrl = `${wsProtocol}//${window.location.host}/api/depth/depth/stream`;
    
    if (model) {
       wsUrl += `?model=${model}`;
    }

    wsConnection = new WebSocket(wsUrl);
    wsConnection.binaryType = 'arraybuffer';

    wsConnection.onopen = () => resolve(wsConnection);
    wsConnection.onerror = (err) => reject(err);
    wsConnection.onclose = () => {
      wsConnection = null;
      while (pendingRejecters.length > 0) {
        pendingRejecters.shift()(new Error('WebSocket closed'));
        pendingResolvers.shift();
      }
    };

    wsConnection.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const data = JSON.parse(event.data);
        if (data.error && pendingRejecters.length > 0) {
          pendingResolvers.shift();
          pendingRejecters.shift()(new Error(data.error));
        } else if (pendingResolvers.length > 0) {
          pendingRejecters.shift();
          pendingResolvers.shift()(data);
        }
      }
    };
  });
}

/**
 * Query the depth service using WebSockets and raw ArrayBuffers for near-zero latency.
 *
 * @param {HTMLVideoElement} video  The video element
 * @param {Array<[number,number]>} points  pixel coords [[x,y], ...] (Must be scaled to 640x360 already)
 * @param {string} [model]  override model name
 * @returns {Promise<{ depths: number[], model: string, inference_ms: number }>}
 */
export async function queryDepthAtPointsStream(video, points, model) {
  const ws = await getWebSocket(model);
  
  if (!captureCanvas) {
    captureCanvas = document.createElement("canvas");
    captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });
    captureCanvas.width = DEPTH_TARGET_WIDTH;
    captureCanvas.height = DEPTH_TARGET_HEIGHT;
  }

  captureCtx.drawImage(video, 0, 0, DEPTH_TARGET_WIDTH, DEPTH_TARGET_HEIGHT);
  
  // Extract raw RGBA bytes
  const imageData = captureCtx.getImageData(0, 0, DEPTH_TARGET_WIDTH, DEPTH_TARGET_HEIGHT);
  const rgba = imageData.data; // Uint8ClampedArray
  
  const numPixels = DEPTH_TARGET_WIDTH * DEPTH_TARGET_HEIGHT;
  
  // Protocol: 
  // [2 bytes: N points] + [N * 4 bytes: coords] + [RGB pixels]
  const N = points.length;
  const headerBytes = 2 + (N * 4);
  const bufferLength = headerBytes + (numPixels * 3);
  
  const arrayBuffer = new ArrayBuffer(bufferLength);
  const dataView = new DataView(arrayBuffer);
  const uint8View = new Uint8Array(arrayBuffer);
  
  // 1. Pack N (Uint16)
  dataView.setUint16(0, N, true);

  // 2. Pack Coordinates (N x 2x Uint16) Little Endian
  let offset = 2;
  for (let i = 0; i < N; i++) {
     dataView.setUint16(offset, points[i][0], true);
     offset += 2;
     dataView.setUint16(offset, points[i][1], true);
     offset += 2;
  }
  
  // 3. Pack RGB pixels (skip alpha channel from RGBA)
  let rgbIdx = headerBytes;
  for (let i = 0; i < rgba.length; i += 4) {
    uint8View[rgbIdx++] = rgba[i];     // R
    uint8View[rgbIdx++] = rgba[i+1];   // G
    uint8View[rgbIdx++] = rgba[i+2];   // B
  }
  
  return new Promise((resolve, reject) => {
    pendingResolvers.push(resolve);
    pendingRejecters.push(reject);
    
    // Send binary
    ws.send(arrayBuffer);
  });
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
