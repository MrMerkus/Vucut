// 1. IMPORT DÜZELTMESİ: vision_bundle.js dosyası açıkça belirtildi.
import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";

let poseLandmarker;
let handLandmarker;
let webcamRunning = false;
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Modelleri yükle
const setupModels = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU" // Tarayıcın GPU'yu desteklemiyorsa burayı "CPU" yapabilirsin
    },
    runningMode: "VIDEO"
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });

  // Sistem hazır olduğunda butonu belirginleştirecek sınıfı ekle
  document.body.classList.add("ready");
};
setupModels();

const webcamButton = document.getElementById("webcamButton");
webcamButton.addEventListener("click", async () => {
  // 2. KORUMA: Modeller henüz yüklenmediyse butonu devre dışı bırak veya uyar
  if (!poseLandmarker || !handLandmarker) {
    alert("Yapay zeka modelleri henüz indiriliyor, lütfen birkaç saniye daha bekleyin.");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    webcamButton.innerText = "KAMERAYI AÇ";
    // Kamerayı tamamen kapat
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  } else {
    webcamRunning = true;
    webcamButton.innerText = "KAMERAYI KAPAT";
    
    // Kameraya erişim isteği
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    
    // 3. EVENT LISTENER DÜZELTMESİ: Üst üste yığılmayı engellemek için onloadeddata kullanımı
    video.onloadeddata = async () => {
      // 4. AUTOPLAY GÜVENLİĞİ: Videoyu açıkça oynatmaya zorla
      await video.play();
      predictWebcam();
    };
  }
});

let lastVideoTime = -1;
async function predictWebcam() {
  // Kamera durdurulduysa animasyon döngüsünden çık
  if (!webcamRunning) return;

  // Koordinat kaymasını önleyen kritik nokta: Canvas boyutunu video ile eşitle
  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  let startTimeMs = performance.now();
  
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Vücut Tahmini
    const poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
    if (poseResults.landmarks) {
      for (const landmark of poseResults.landmarks) {
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { color: "#00bfa5", lineWidth: 2 });
        drawingUtils.drawLandmarks(landmark, { color: "#fff", radius: 1 });
      }
    }

    // El Tahmini
    const handResults = handLandmarker.detectForVideo(video, startTimeMs);
    if (handResults.landmarks) {
      for (const landmarks of handResults.landmarks) {
        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#ff4081", lineWidth: 3 });
        drawingUtils.drawLandmarks(landmarks, { color: "#fff", radius: 2 });
      }
    }
    canvasCtx.restore();
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}
