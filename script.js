import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
 
// ─── DOM referansları ───────────────────────────────────────────────────────
const video         = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx     = canvasElement.getContext("2d");
const statusEl      = document.querySelector('.status');
const webcamButton  = document.getElementById("webcamButton");
 
// ─── Sabitler ──────────────────────────────────────────────────────────────
const VISIBILITY_THRESHOLD = 0.60;
 
/**
 * Temporal (zaman-tabanlı) yumuşatma filtresi.
 * Her landmark'ı önceki N frame ile ağırlıklı ortalama yaparak titreyi keser.
 *
 * alpha = [0..1]:  1 → filtre yok (ham),  0 → hiç hareket etmez.
 * 0.5 dengeli bir başlangıç noktasıdır; hızlı hareketler için artırın.
 */
const SMOOTH_ALPHA = 0.50;
 
// Pose ve el için ayrı smoothing durumu
const poseSmooth = [];   // poseSmooth[poseIdx][landmarkIdx] = {x,y,z}
const handSmooth = [];   // handSmooth[handIdx][landmarkIdx]  = {x,y,z}
 
function smoothLandmarks(prevCache, cacheIdx, rawLandmarks) {
  if (!prevCache[cacheIdx]) {
    // İlk kare: doğrudan kopyala
    prevCache[cacheIdx] = rawLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
    return prevCache[cacheIdx];
  }
 
  const prev = prevCache[cacheIdx];
  const smoothed = rawLandmarks.map((lm, i) => ({
    x: SMOOTH_ALPHA * lm.x + (1 - SMOOTH_ALPHA) * prev[i].x,
    y: SMOOTH_ALPHA * lm.y + (1 - SMOOTH_ALPHA) * prev[i].y,
    z: SMOOTH_ALPHA * lm.z + (1 - SMOOTH_ALPHA) * prev[i].z,
    visibility: lm.visibility   // visibility smooth edilmez (kararlılığı bozar)
  }));
 
  prevCache[cacheIdx] = smoothed;
  return smoothed;
}
 
// ─── Model kurulumu ────────────────────────────────────────────────────────
let poseLandmarker;
let handLandmarker;
let modelsReady = false;
 
const setupModels = async () => {
  try {
    statusEl.textContent = "Modeller yükleniyor…";
 
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
 
    // VÜCUT TAKİBİ — full model, GPU delegasyonu
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 2,
      minPoseDetectionConfidence: 0.70,
      minPosePresenceConfidence: 0.70,
      minTrackingConfidence: 0.75
    });
 
    // EL TAKİBİ — GPU delegasyonu
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.65,
      minHandPresenceConfidence: 0.65,
      minTrackingConfidence: 0.70
    });
 
    modelsReady = true;
    statusEl.textContent = "GPU Sistem Aktif. Kamerayı açabilirsiniz.";
    webcamButton.disabled = false;
  } catch (err) {
    statusEl.textContent = "Model yükleme hatası: " + err.message;
    console.error(err);
  }
};
 
webcamButton.disabled = true;
setupModels();
 
// ─── Kamera kontrolü ───────────────────────────────────────────────────────
let webcamRunning = false;
 
webcamButton.addEventListener("click", async () => {
  if (!modelsReady) return;
 
  if (webcamRunning) {
    webcamRunning = false;
    webcamButton.querySelector('.mdc-button__label').textContent = "KAMERAYI AÇ";
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    // Smoothing önbelleklerini sıfırla
    poseSmooth.length = 0;
    handSmooth.length = 0;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    return;
  }
 
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
    });
    video.srcObject = stream;
    webcamRunning = true;
    webcamButton.querySelector('.mdc-button__label').textContent = "KAMERAYI KAPAT";
 
    // Video metadata yüklenince canvas boyutunu ayarla, sonra döngüyü başlat
    video.addEventListener("loadedmetadata", () => {
      canvasElement.width  = video.videoWidth;
      canvasElement.height = video.videoHeight;
    }, { once: true });
 
    video.addEventListener("loadeddata", () => {
      requestAnimationFrame(predictWebcam);
    }, { once: true });
 
  } catch (err) {
    statusEl.textContent = "Kamera erişim hatası: " + err.message;
    webcamRunning = false;
  }
});
 
// ─── Çizim yardımcıları ────────────────────────────────────────────────────
const drawingUtils = new DrawingUtils(canvasCtx);
 
/**
 * Canvas'a çizim yaparken aynalama (mirror) düzeltmesi uygular.
 *
 * Video CSS'te `rotateY(180deg)` ile aynalı gösteriliyor.
 * Canvas'taki koordinatları da aynalamamız gerekir; aksi hâlde
 * iskelet video üzerinde kayar.
 *
 * Çözüm: ctx'i yatayda çevir → çizim yap → geri döndür.
 */
function withMirror(fn) {
  canvasCtx.save();
  // Merkezi pivot noktası olarak kullan
  canvasCtx.translate(canvasElement.width, 0);
  canvasCtx.scale(-1, 1);
  fn();
  canvasCtx.restore();
}
 
/** Normalised [0,1] koordinatı → pixel koordinatı */
function toPixel(landmark) {
  return {
    px: landmark.x * canvasElement.width,
    py: landmark.y * canvasElement.height
  };
}
 
// ─── Ana tahmin döngüsü ────────────────────────────────────────────────────
let lastVideoTime = -1;
 
async function predictWebcam() {
  if (!webcamRunning) return;
 
  // Canvas boyutunu video ile senkronize tut (yalnızca değişince)
  if (canvasElement.width !== video.videoWidth && video.videoWidth > 0) {
    canvasElement.width  = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }
 
  const nowMs = performance.now();
 
  // Aynı frame'i iki kez işleme
  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;
 
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
 
    // ── 1. VÜCUT TAKİBİ ─────────────────────────────────────────────────
    const poseResults = poseLandmarker.detectForVideo(video, nowMs);
 
    if (poseResults.landmarks && poseResults.landmarks.length > 0) {
      poseResults.landmarks.forEach((rawPose, poseIdx) => {
        // Temporal smoothing uygula
        const pose = smoothLandmarks(poseSmooth, poseIdx, rawPose);
 
        withMirror(() => {
          // Bağlantı çizgileri
          PoseLandmarker.POSE_CONNECTIONS.forEach(({ start, end }) => {
            const s = pose[start];
            const e = pose[end];
 
            const sVis = s.visibility ?? 1;
            const eVis = e.visibility ?? 1;
            if (sVis < VISIBILITY_THRESHOLD || eVis < VISIBILITY_THRESHOLD) return;
 
            const { px: sx, py: sy } = toPixel(s);
            const { px: ex, py: ey } = toPixel(e);
 
            canvasCtx.beginPath();
            canvasCtx.moveTo(sx, sy);
            canvasCtx.lineTo(ex, ey);
            canvasCtx.strokeStyle = "rgba(0, 191, 165, 0.85)";
            canvasCtx.lineWidth = 3;
            canvasCtx.lineJoin = "round";
            canvasCtx.stroke();
          });
 
          // Eklem noktaları
          pose.forEach(lm => {
            const vis = lm.visibility ?? 1;
            if (vis < VISIBILITY_THRESHOLD) return;
 
            const { px, py } = toPixel(lm);
            canvasCtx.beginPath();
            canvasCtx.arc(px, py, 5, 0, 2 * Math.PI);
            canvasCtx.fillStyle   = "#ffffff";
            canvasCtx.strokeStyle = "rgba(0, 191, 165, 0.9)";
            canvasCtx.lineWidth   = 2;
            canvasCtx.fill();
            canvasCtx.stroke();
          });
        });
      });
 
      // Algılanmayan pose önbelleklerini temizle (kişi sahneden çıkınca)
      if (poseResults.landmarks.length < poseSmooth.length) {
        poseSmooth.splice(poseResults.landmarks.length);
      }
    } else {
      poseSmooth.length = 0;
    }
 
    // ── 2. EL TAKİBİ ────────────────────────────────────────────────────
    // El tahmini için 1 ms ileri zaman damgası — aynı timestamp çakışmasını önler
    const handResults = handLandmarker.detectForVideo(video, nowMs + 1);
 
    if (handResults.landmarks && handResults.landmarks.length > 0) {
      handResults.landmarks.forEach((rawHand, handIdx) => {
        const hand = smoothLandmarks(handSmooth, handIdx, rawHand);
 
        withMirror(() => {
          // Bağlantılar
          HandLandmarker.HAND_CONNECTIONS.forEach(({ start, end }) => {
            const s = hand[start];
            const e = hand[end];
            const { px: sx, py: sy } = toPixel(s);
            const { px: ex, py: ey } = toPixel(e);
 
            canvasCtx.beginPath();
            canvasCtx.moveTo(sx, sy);
            canvasCtx.lineTo(ex, ey);
            canvasCtx.strokeStyle = "rgba(255, 64, 129, 0.85)";
            canvasCtx.lineWidth   = 2.5;
            canvasCtx.lineJoin    = "round";
            canvasCtx.stroke();
          });
 
          // Eklem noktaları
          hand.forEach(lm => {
            const { px, py } = toPixel(lm);
            canvasCtx.beginPath();
            canvasCtx.arc(px, py, 4, 0, 2 * Math.PI);
            canvasCtx.fillStyle   = "#ffffff";
            canvasCtx.strokeStyle = "rgba(255, 64, 129, 0.9)";
            canvasCtx.lineWidth   = 1.5;
            canvasCtx.fill();
            canvasCtx.stroke();
          });
        });
      });
 
      if (handResults.landmarks.length < handSmooth.length) {
        handSmooth.splice(handResults.landmarks.length);
      }
    } else {
      handSmooth.length = 0;
    }
  }
 
  requestAnimationFrame(predictWebcam);
}
 
