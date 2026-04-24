import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";

// ── DOM ──────────────────────────────────────────────────────────────────────
const video         = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx     = canvasElement.getContext("2d");
const drawingUtils  = new DrawingUtils(canvasCtx);

const webcamButton    = document.getElementById("webcamButton");
const btnText         = document.getElementById("btnText");
const screenshotBtn   = document.getElementById("screenshotButton");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const modelStatus     = document.getElementById("model-status");
const fpsDisplay      = document.getElementById("fps-display");

// ── STATE ─────────────────────────────────────────────────────────────────────
let poseLandmarker, handLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;

// FPS tracking
let fpsFrameCount = 0;
let fpsLastTime   = performance.now();

// ── MODEL SETUP ───────────────────────────────────────────────────────────────
const setupModels = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    minPoseDetectionConfidence: 0.65,
    minPosePresenceConfidence: 0.65,
    minTrackingConfidence: 0.65
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.65,
    minHandPresenceConfidence: 0.65,
    minTrackingConfidence: 0.65
  });

  modelStatus.textContent = "● MODEL HAZIR";
  modelStatus.classList.add("ready");
};
setupModels();

// ── ANGLE MATH ────────────────────────────────────────────────────────────────
/**
 * Üç nokta arasındaki açıyı derece cinsinden hesaplar.
 * B noktası köşe noktasıdır (A–B–C üçgeni).
 */
function calcAngle(A, B, C) {
  const BA = { x: A.x - B.x, y: A.y - B.y };
  const BC = { x: C.x - B.x, y: C.y - B.y };
  const dot  = BA.x * BC.x + BA.y * BC.y;
  const magA = Math.hypot(BA.x, BA.y);
  const magC = Math.hypot(BC.x, BC.y);
  if (magA === 0 || magC === 0) return null;
  const cos  = Math.max(-1, Math.min(1, dot / (magA * magC)));
  return Math.round(Math.acos(cos) * (180 / Math.PI));
}

// ── UI HELPERS ────────────────────────────────────────────────────────────────
function updateAngleCard(id, degrees) {
  const valueEl = document.getElementById(`angle-${id}`);
  const barEl   = document.getElementById(`bar-${id}`);
  const card    = document.getElementById(`card-${id}`);

  if (degrees === null) {
    valueEl.textContent = "—°";
    barEl.style.width = "0%";
    card.classList.remove("active");
    return;
  }

  valueEl.textContent = `${degrees}°`;
  // Bar: 0°=0%, 180°=100%
  const pct = Math.min(100, (degrees / 180) * 100);
  barEl.style.width = `${pct}%`;
  card.classList.add("active");

  // Renk: tam açılmış (yakın 180°) → teal, bükümlü (0–90°) → pink
  const hue = degrees < 90 ? "var(--pink)" : degrees < 150 ? "var(--yellow)" : "var(--teal)";
  barEl.style.background = hue;
  valueEl.style.color    = hue;
}

function updateStat(id, isActive, label = null) {
  const el = document.getElementById(id);
  el.textContent = isActive ? (label || "EVET") : "HAYIR";
  el.className = "stat-value" + (isActive ? " yes" : "");
}

function updateSymmetry(score) {
  const el  = document.getElementById("symmetry-score");
  const bar = document.getElementById("symmetry-bar");
  if (score === null) {
    el.textContent = "—";
    bar.style.width = "0%";
    return;
  }
  el.textContent = `${score}%`;
  bar.style.width = `${score}%`;
  // Renk kodu
  el.style.color = score > 80 ? "var(--green)" : score > 50 ? "var(--yellow)" : "var(--red)";
}

// ── CANVAS ANGLE OVERLAY ──────────────────────────────────────────────────────
/**
 * Canvas üzerine açı etiketini yazar.
 * x, y: normalize edilmiş koordinatlar (0-1 arası).
 */
function drawAngleLabel(ctx, x, y, degrees, color = "#00e5cc") {
  if (degrees === null) return;
  const cx = (1 - x) * canvasElement.width;  // aynalı x
  const cy = y       * canvasElement.height;

  ctx.save();
  ctx.font = "bold 13px 'Space Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Arka plan pill
  const text  = `${degrees}°`;
  const tw    = ctx.measureText(text).width + 10;
  const th    = 18;

  ctx.fillStyle = "rgba(5,10,14,0.75)";
  ctx.beginPath();
  ctx.roundRect(cx - tw / 2, cy - th / 2, tw, th, 4);
  ctx.fill();

  // Çerçeve
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1;
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.shadowColor  = color;
  ctx.shadowBlur   = 6;
  ctx.fillText(text, cx, cy);
  ctx.restore();
}

// ── WEBCAM TOGGLE ─────────────────────────────────────────────────────────────
webcamButton.addEventListener("click", async () => {
  if (!poseLandmarker || !handLandmarker) {
    alert("Yapay zeka modelleri henüz yükleniyor, lütfen birkaç saniye bekleyin.");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    btnText.textContent = "KAMERAYI AÇ";
    webcamButton.classList.remove("active");
    screenshotBtn.disabled = true;
    video.srcObject?.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    cameraPlaceholder.classList.remove("hidden");

    // Metrikleri sıfırla
    ["left-elbow","right-elbow","left-shoulder","right-shoulder","left-knee","right-knee","shoulder-spread","hip"]
      .forEach(id => updateAngleCard(id, null));
    updateStat("stat-pose", false);
    updateStat("stat-left-hand", false);
    updateStat("stat-right-hand", false);
    document.getElementById("stat-landmarks").textContent = "0";
    updateSymmetry(null);
    fpsDisplay.textContent = "0";
  } else {
    webcamRunning = true;
    btnText.textContent = "KAMERAYI KAPAT";
    webcamButton.classList.add("active");
    screenshotBtn.disabled = false;
    cameraPlaceholder.classList.add("hidden");

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
    });
    video.srcObject = stream;
    video.onloadeddata = async () => {
      await video.play();
      predictWebcam();
    };
  }
});

// ── SCREENSHOT ────────────────────────────────────────────────────────────────
screenshotBtn.addEventListener("click", () => {
  // video + canvas birleştir
  const tmp = document.createElement("canvas");
  tmp.width  = canvasElement.width;
  tmp.height = canvasElement.height;
  const tCtx = tmp.getContext("2d");

  // Aynalı video frame
  tCtx.save();
  tCtx.translate(tmp.width, 0);
  tCtx.scale(-1, 1);
  tCtx.drawImage(video, 0, 0, tmp.width, tmp.height);
  tCtx.restore();

  // canvas çizimlerini ekle (zaten aynalı)
  tCtx.save();
  tCtx.translate(tmp.width, 0);
  tCtx.scale(-1, 1);
  tCtx.drawImage(canvasElement, 0, 0);
  tCtx.restore();

  const link = document.createElement("a");
  link.download = `hareket-analiz-${Date.now()}.png`;
  link.href = tmp.toDataURL("image/png");
  link.click();
});

// ── MAIN LOOP ─────────────────────────────────────────────────────────────────
async function predictWebcam() {
  if (!webcamRunning) return;

  // Canvas boyutunu eşitle
  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width  = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  const now = performance.now();

  // FPS hesapla
  fpsFrameCount++;
  if (now - fpsLastTime >= 500) {
    fpsDisplay.textContent = Math.round(fpsFrameCount * 1000 / (now - fpsLastTime));
    fpsFrameCount = 0;
    fpsLastTime   = now;
  }

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    const poseResults = poseLandmarker.detectForVideo(video, now);
    const handResults = handLandmarker.detectForVideo(video, now);

    // ── POSE ────────────────────────────────────────────────────────────────
    const hasPose = poseResults.landmarks && poseResults.landmarks.length > 0;
    updateStat("stat-pose", hasPose, "ALGILANDI");

    if (hasPose) {
      const lm = poseResults.landmarks[0]; // ilk kişi
      document.getElementById("stat-landmarks").textContent =
        (lm.length + (handResults.landmarks ? handResults.landmarks.flat().length : 0));

      // Bağlantı & nokta çizimi
      canvasCtx.shadowColor = "#00e5cc";
      canvasCtx.shadowBlur  = 12;
      drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: "#00bfa5", lineWidth: 3 });
      drawingUtils.drawLandmarks(lm, { color: "#ffffff", fillColor: "#00bfa5", lineWidth: 1, radius: 4 });
      canvasCtx.shadowBlur = 0;

      // ── AÇI HESAPLARI ────────────────────────────────────────────────────
      // MediaPipe Pose landmarK indeksleri:
      // 11=sol omuz  12=sağ omuz  13=sol dirsek  14=sağ dirsek
      // 15=sol bilek 16=sağ bilek 23=sol kalça  24=sağ kalça
      // 25=sol diz   26=sağ diz   27=sol ayak   28=sağ ayak

      const leftElbow      = calcAngle(lm[11], lm[13], lm[15]); // omuz-dirsek-bilek
      const rightElbow     = calcAngle(lm[12], lm[14], lm[16]);
      const leftShoulder   = calcAngle(lm[13], lm[11], lm[23]); // dirsek-omuz-kalça
      const rightShoulder  = calcAngle(lm[14], lm[12], lm[24]);
      const leftKnee       = calcAngle(lm[23], lm[25], lm[27]); // kalça-diz-ayak
      const rightKnee      = calcAngle(lm[24], lm[26], lm[28]);
      const shoulderSpread = calcAngle(lm[23], lm[11], lm[12]); // sol kalça-sol omuz-sağ omuz
      const hipAngle       = calcAngle(lm[11], lm[23], lm[25]); // omuz-kalça-diz

      updateAngleCard("left-elbow",      leftElbow);
      updateAngleCard("right-elbow",     rightElbow);
      updateAngleCard("left-shoulder",   leftShoulder);
      updateAngleCard("right-shoulder",  rightShoulder);
      updateAngleCard("left-knee",       leftKnee);
      updateAngleCard("right-knee",      rightKnee);
      updateAngleCard("shoulder-spread", shoulderSpread);
      updateAngleCard("hip",             hipAngle);

      // Canvas üzerine açı etiketleri
      if (leftElbow  !== null) drawAngleLabel(canvasCtx, lm[13].x, lm[13].y, leftElbow,  "#00e5cc");
      if (rightElbow !== null) drawAngleLabel(canvasCtx, lm[14].x, lm[14].y, rightElbow, "#00e5cc");
      if (leftKnee   !== null) drawAngleLabel(canvasCtx, lm[25].x, lm[25].y, leftKnee,   "#ffd600");
      if (rightKnee  !== null) drawAngleLabel(canvasCtx, lm[26].x, lm[26].y, rightKnee,  "#ffd600");

      // ── SİMETRİ SKORU ────────────────────────────────────────────────────
      const pairs = [
        [leftElbow,  rightElbow],
        [leftShoulder, rightShoulder],
        [leftKnee,   rightKnee]
      ].filter(([a, b]) => a !== null && b !== null);

      if (pairs.length > 0) {
        const avgDiff = pairs.reduce((sum, [a, b]) => sum + Math.abs(a - b), 0) / pairs.length;
        // 0 fark → %100 simetri, 90° fark → %0
        const score = Math.round(Math.max(0, 100 - (avgDiff / 90) * 100));
        updateSymmetry(score);
      } else {
        updateSymmetry(null);
      }
    } else {
      // Pose yok → sıfırla
      ["left-elbow","right-elbow","left-shoulder","right-shoulder","left-knee","right-knee","shoulder-spread","hip"]
        .forEach(id => updateAngleCard(id, null));
      document.getElementById("stat-landmarks").textContent = "0";
      updateSymmetry(null);
    }

    // ── HANDS ────────────────────────────────────────────────────────────────
    let leftHandDetected  = false;
    let rightHandDetected = false;

    if (handResults.landmarks && handResults.landmarks.length > 0) {
      for (let i = 0; i < handResults.landmarks.length; i++) {
        const landmarks  = handResults.landmarks[i];
        const handedness = handResults.handedness?.[i]?.[0]?.categoryName;

        // Not: video aynalı, el etiketleri de tersine döner
        if (handedness === "Right") leftHandDetected  = true;
        if (handedness === "Left")  rightHandDetected = true;

        canvasCtx.shadowColor = "#ff4081";
        canvasCtx.shadowBlur  = 12;
        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#ff4081", lineWidth: 2.5 });
        drawingUtils.drawLandmarks(landmarks, { color: "#ffffff", fillColor: "#ff4081", lineWidth: 1, radius: 3 });
        canvasCtx.shadowBlur = 0;
      }
    }

    updateStat("stat-left-hand",  leftHandDetected,  "ALGILANDI");
    updateStat("stat-right-hand", rightHandDetected, "ALGILANDI");

    canvasCtx.restore();
  }

  if (webcamRunning) requestAnimationFrame(predictWebcam);
}
