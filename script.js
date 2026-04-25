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

const webcamButton      = document.getElementById("webcamButton");
const btnText           = document.getElementById("btnText");
const screenshotBtn     = document.getElementById("screenshotButton");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const modelStatus       = document.getElementById("model-status");
const fpsDisplay        = document.getElementById("fps-display");
const metricsContainer  = document.getElementById("metrics-container");

// ── STATE ─────────────────────────────────────────────────────────────────────
let poseLandmarker, handLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;
let fpsFrameCount = 0;
let fpsLastTime   = performance.now();
let activePersons = 0; // Şu an ekranda takip edilen kişi sayısı

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
    numPoses: 4, // AYAR: Maksimum aynı anda algılanacak kişi sayısı
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
    numHands: 8, // AYAR: 4 kişi x 2 el = 8 el
    minHandDetectionConfidence: 0.65,
    minHandPresenceConfidence: 0.65,
    minTrackingConfidence: 0.65
  });

  modelStatus.textContent = "● MODEL HAZIR";
  modelStatus.classList.add("ready");
};
setupModels();

// ── AÇI HESAPLAMA ─────────────────────────────────────────────────────────────
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

function drawAngleLabel(ctx, x, y, degrees, color = "#00e5cc") {
  if (degrees === null) return;
  const cx = (1 - x) * canvasElement.width;
  const cy = y       * canvasElement.height;

  ctx.save();
  ctx.font = "bold 13px 'Space Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const text  = `${degrees}°`;
  const tw    = ctx.measureText(text).width + 10;
  const th    = 18;

  ctx.fillStyle = "rgba(5,10,14,0.75)";
  ctx.beginPath();
  ctx.roundRect(cx - tw / 2, cy - th / 2, tw, th, 4);
  ctx.fill();

  ctx.strokeStyle = color;
  ctx.lineWidth   = 1;
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.shadowColor  = color;
  ctx.shadowBlur   = 6;
  ctx.fillText(text, cx, cy);
  ctx.restore();
}

// ── DİNAMİK ARAYÜZ (UI) YÖNETİMİ ──────────────────────────────────────────────
// Yeni biri algılandığında onun için HTML iskeleti oluşturur
function buildPersonUI(pIndex) {
  const wrapper = document.createElement('div');
  wrapper.className = 'person-wrapper';
  wrapper.id = `person-wrapper-${pIndex}`;
  
  // Renk kodlaması (kişileri birbirinden ayırmak için)
  const colors = ['var(--teal)', 'var(--pink)', 'var(--yellow)', 'var(--green)'];
  const pColor = colors[pIndex % colors.length];
  wrapper.style.borderLeftColor = pColor;

  wrapper.innerHTML = `
    <div class="person-header" style="color: ${pColor}">
      <span>DENEK #${pIndex + 1}</span>
      <span class="person-id-badge" style="background: ${pColor}">AKTİF</span>
    </div>
    
    <div class="panel-section">
      <h3 class="panel-title">EKLEM AÇILARI</h3>
      <div class="angles-grid">
        ${['left-elbow', 'right-elbow', 'left-shoulder', 'right-shoulder', 'left-knee', 'right-knee'].map(joint => `
          <div class="angle-card" id="card-${pIndex}-${joint}">
            <div class="angle-label">${joint.replace('-', ' ').toUpperCase()}</div>
            <div class="angle-value" id="angle-${pIndex}-${joint}">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="bar-${pIndex}-${joint}"></div></div>
          </div>
        `).join('')}
      </div>
    </div>

    <div class="panel-section" style="margin-top: 12px;">
      <h3 class="panel-title">SİMETRİ SKORU</h3>
      <div class="symmetry-display">
        <div class="symmetry-score" id="symmetry-score-${pIndex}">—</div>
        <div class="symmetry-bar-wrap">
          <div class="symmetry-bar" id="symmetry-bar-${pIndex}"></div>
        </div>
      </div>
    </div>
  `;
  return wrapper;
}

// Kişinin verilerini sadece DOM üzerinde günceller (Performans için)
function updatePersonData(pIndex, angles, score) {
  // Açıları güncelle
  for (const [key, value] of Object.entries(angles)) {
    const valueEl = document.getElementById(`angle-${pIndex}-${key}`);
    const barEl   = document.getElementById(`bar-${pIndex}-${key}`);
    const card    = document.getElementById(`card-${pIndex}-${key}`);

    if (value === null) {
      if(valueEl) valueEl.textContent = "—°";
      if(barEl) barEl.style.width = "0%";
      if(card) card.classList.remove("active");
      continue;
    }

    if(valueEl) valueEl.textContent = `${value}°`;
    const pct = Math.min(100, (value / 180) * 100);
    
    if(barEl) {
      barEl.style.width = `${pct}%`;
      const hue = value < 90 ? "var(--pink)" : value < 150 ? "var(--yellow)" : "var(--teal)";
      barEl.style.background = hue;
      valueEl.style.color    = hue;
    }
    if(card) card.classList.add("active");
  }

  // Simetriyi güncelle
  const symEl  = document.getElementById(`symmetry-score-${pIndex}`);
  const symBar = document.getElementById(`symmetry-bar-${pIndex}`);
  if (score === null) {
    if(symEl) symEl.textContent = "—";
    if(symBar) symBar.style.width = "0%";
  } else {
    if(symEl) {
      symEl.textContent = `${score}%`;
      symEl.style.color = score > 80 ? "var(--green)" : score > 50 ? "var(--yellow)" : "var(--red)";
    }
    if(symBar) symBar.style.width = `${score}%`;
  }
}

// ── WEBCAM KONTROLÜ ───────────────────────────────────────────────────────────
webcamButton.addEventListener("click", async () => {
  if (!poseLandmarker || !handLandmarker) {
    alert("Modeller yükleniyor, lütfen bekleyin.");
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
    
    metricsContainer.innerHTML = ""; // Paneli temizle
    activePersons = 0;
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

// ── ANA DÖNGÜ (MAIN LOOP) ─────────────────────────────────────────────────────
async function predictWebcam() {
  if (!webcamRunning) return;

  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width  = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  const now = performance.now();
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
    
    // UI'daki kişi kartı sayısını senkronize et
    const detectedPersonsCount = poseResults.landmarks ? poseResults.landmarks.length : 0;
    if (detectedPersonsCount !== activePersons) {
      metricsContainer.innerHTML = ""; // Kişi sayısı değiştiyse paneli sıfırla
      for (let i = 0; i < detectedPersonsCount; i++) {
        metricsContainer.appendChild(buildPersonUI(i));
      }
      activePersons = detectedPersonsCount;
    }

    // ── HER BİR KİŞİ İÇİN DÖNGÜ ──────────────────────────────────────────────
    if (poseResults.landmarks) {
      poseResults.landmarks.forEach((lm, pIndex) => {
        // Renk kodlaması
        const colors = ['#00e5cc', '#ff4081', '#ffd600', '#69ff47'];
        const pColor = colors[pIndex % colors.length];

        // İskelet Çizimi
        canvasCtx.shadowColor = pColor;
        canvasCtx.shadowBlur  = 8;
        drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: pColor, lineWidth: 3 });
        drawingUtils.drawLandmarks(lm, { color: "#ffffff", fillColor: pColor, lineWidth: 1, radius: 4 });
        canvasCtx.shadowBlur = 0;

        // Açı Hesaplamaları
        const angles = {
          'left-elbow': calcAngle(lm[11], lm[13], lm[15]),
          'right-elbow': calcAngle(lm[12], lm[14], lm[16]),
          'left-shoulder': calcAngle(lm[13], lm[11], lm[23]),
          'right-shoulder': calcAngle(lm[14], lm[12], lm[24]),
          'left-knee': calcAngle(lm[23], lm[25], lm[27]),
          'right-knee': calcAngle(lm[24], lm[26], lm[28])
        };

        // Canvas üzerine etiketleri yazdır (Sadece kollar ve dizler)
        if (angles['left-elbow'] !== null) drawAngleLabel(canvasCtx, lm[13].x, lm[13].y, angles['left-elbow'], pColor);
        if (angles['right-elbow'] !== null) drawAngleLabel(canvasCtx, lm[14].x, lm[14].y, angles['right-elbow'], pColor);
        if (angles['left-knee'] !== null) drawAngleLabel(canvasCtx, lm[25].x, lm[25].y, angles['left-knee'], pColor);
        if (angles['right-knee'] !== null) drawAngleLabel(canvasCtx, lm[26].x, lm[26].y, angles['right-knee'], pColor);

        // Simetri Skoru
        const pairs = [
          [angles['left-elbow'], angles['right-elbow']],
          [angles['left-shoulder'], angles['right-shoulder']],
          [angles['left-knee'], angles['right-knee']]
        ].filter(([a, b]) => a !== null && b !== null);

        let score = null;
        if (pairs.length > 0) {
          const avgDiff = pairs.reduce((sum, [a, b]) => sum + Math.abs(a - b), 0) / pairs.length;
          score = Math.round(Math.max(0, 100 - (avgDiff / 90) * 100));
        }

        // Sağ paneldeki spesifik kişiyi güncelle
        updatePersonData(pIndex, angles, score);
      });
    }

    // ── ELLER İÇİN DÖNGÜ (Tüm elleri ekrana çiz) ───────────────────────────
    if (handResults.landmarks && handResults.landmarks.length > 0) {
      for (let i = 0; i < handResults.landmarks.length; i++) {
        const lm = handResults.landmarks[i];
        canvasCtx.shadowColor = "#ff4081";
        canvasCtx.shadowBlur  = 8;
        drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#ff4081", lineWidth: 2 });
        drawingUtils.drawLandmarks(lm, { color: "#ffffff", fillColor: "#ff4081", lineWidth: 1, radius: 3 });
        canvasCtx.shadowBlur = 0;
      }
    }

    canvasCtx.restore();
  }

  if (webcamRunning) requestAnimationFrame(predictWebcam);
}
