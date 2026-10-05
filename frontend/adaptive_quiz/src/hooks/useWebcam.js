/**
 * hooks/useWebcam.js
 * Custom hook that continuously captures webcam frames in the background.
 * Periodically POSTs JPEG frames to POST /api/detect-emotion (Python backend using model/best.pt).
 * Returns: { videoRef, currentExpression, isWebcamActive, error }
 *
 * The webcam runs silently in the background the entire time the assessment
 * is active. The latest detected expression is always available via
 * `currentExpression` for use when submitting each answer.
 *
 * Env: VITE_API_URL (default …/api), or set VITE_DETECT_EMOTION_URL to the full detect URL.
 * To use the standalone ML service instead: VITE_DETECT_EMOTION_URL=http://localhost:8000/detect-emotion
 */
import { useEffect, useRef, useState, useCallback } from 'react';

// Default: Python STEM API — POST …/api/detect-emotion (uses backend/model/best.pt)
const API_BASE = import.meta.env.VITE_API_URL || '/api/adaptive-quiz';
const EMOTION_DETECT_URL =
  import.meta.env.VITE_DETECT_EMOTION_URL ||
  `${API_BASE.replace(/\/$/, '')}/detect-emotion`;

// How often (ms) to sample a frame and call the emotion endpoint.
// Inference is light (~25-30 ms/frame) and requests are single-flighted, so we can
// poll fast for low-latency feedback; a tick is skipped while one is still in flight.
const CAPTURE_INTERVAL_MS  = 600;
const EXPRESSION_BUFFER_SIZE = 4;
const MIN_AGREEMENT          = 2;
const MIN_CONFIDENCE         = 0.12;

const useWebcam = (isActive = true) => {
  const videoRef          = useRef(null);
  const canvasRef         = useRef(null);
  const intervalRef       = useRef(null);
  const streamRef         = useRef(null);
  const expressionBuffer  = useRef([]); // rolling window for smoothing
  const inFlightRef       = useRef(false); // skip a tick if the last request is still pending

  const [currentExpression, setCurrentExpression] = useState('neutral');
  const [isWebcamActive,    setIsWebcamActive]     = useState(false);
  const [error,             setError]              = useState(null);

  // ── Start webcam stream ────────────────────────────────────────────────────
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsWebcamActive(true);
      setError(null);
    } catch (err) {
      console.warn('⚠️  Webcam unavailable:', err.message);
      setError(err.message);
      setIsWebcamActive(false);
    }
  }, []);

  // ── Capture a frame and detect emotion ────────────────────────────────────
  const captureAndDetect = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    if (inFlightRef.current) return; // previous frame still being analysed — skip
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frameB64 = canvas.toDataURL('image/jpeg', 0.92).split(',')[1];

    inFlightRef.current = true;
    try {
      // POST frame to FastAPI /detect-emotion (abort if it takes too long)
      const token = localStorage.getItem('stemToken');
      const ctrl = new AbortController();
      const kill = setTimeout(() => ctrl.abort(), 4000);
      const res = await fetch(EMOTION_DETECT_URL, {
        method:  'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body:    JSON.stringify({ frame: frameB64 }),
        signal:  ctrl.signal,
      });
      clearTimeout(kill);
      if (res.ok) {
        const data       = await res.json();
        const rawExpr    = data.detectedExpression || 'neutral';
        const confidence = typeof data.confidence === 'number' ? data.confidence : 1.0;

        // Reject low-confidence frames — noisy predictions degrade behavioral signals
        if (confidence < MIN_CONFIDENCE) return;

        // Rolling window: prefer a repeating smile/frown over a mixed-in neutral frame.
        const buf = expressionBuffer.current;
        buf.push(rawExpr);
        if (buf.length > EXPRESSION_BUFFER_SIZE) buf.shift();

        const counts = buf.reduce((acc, e) => { acc[e] = (acc[e] || 0) + 1; return acc; }, {});
        const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const [dominant, votes] = ranked[0];
        const affect = ranked.find(([e]) => e !== 'neutral');
        // Prefer a repeating smile/frown even if one frame in the window is neutral.
        if (affect && affect[1] >= MIN_AGREEMENT) {
          setCurrentExpression(affect[0]);
        } else if (votes >= MIN_AGREEMENT) {
          setCurrentExpression(dominant);
        }
      } else {
        console.warn(`[useWebcam] /detect-emotion returned ${res.status}; keeping last expression`);
      }
    } catch (err) {
      // If ML service is down / slow, keep last known expression — don't crash
      console.warn('[useWebcam] frame capture failed:', err.message);
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  // ── Stop webcam stream ─────────────────────────────────────────────────────
  const stopWebcam = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setIsWebcamActive(false);
  }, []);

  // ── Lifecycle: start/stop based on isActive prop ──────────────────────────
  useEffect(() => {
    // Create off-screen canvas for frame capture
    canvasRef.current = document.createElement('canvas');

    if (!isActive) {
      stopWebcam();
      return undefined;
    }

    let cancelled = false;

    startWebcam().then(() => {
      if (cancelled) return;
      captureAndDetect(); // first sample right away — don't wait a full interval
      intervalRef.current = setInterval(captureAndDetect, CAPTURE_INTERVAL_MS);
    });

    return () => {
      cancelled = true;
      stopWebcam();
    };
  }, [isActive, startWebcam, stopWebcam, captureAndDetect]);

  return { videoRef, currentExpression, isWebcamActive, error };
};

export default useWebcam;
