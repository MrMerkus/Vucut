import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");
let poseLandmarker;
let handLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;

// Modelleri yükle
const initDetectors = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  
  // Vücut Takipçisi
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 2
  });

  // El Takipçisi
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numHands: 2
  });

  demosSection.classList.remove("invisible");
};
initDetectors();

// --- Görsel Tıklama İşlemi ---
const imageContainers = document.getElementsByClassName("detectOnClick");
for (let i = 0; i < imageContainers.length; i++) {
  imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
  if (!poseLandmarker || !handLandmarker) return;

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await poseLandmarker.setOptions({ runningMode: "IMAGE" });
    await handLandmarker.setOptions({ runningMode: "IMAGE" });
  }

  const parent = event.target.parentNode;
  const oldCanvas = parent.getElementsByClassName("canvas");
  for (let i = oldCanvas.length - 1; i >= 0; i--) oldCanvas[i].remove();

  const canvas = document.createElement("canvas");
  canvas.setAttribute("class", "canvas");
  canvas.width = event.target.naturalWidth;
  canvas.height = event.target.naturalHeight;
  canvas.style.width = `${event.target.width}px`;
  canvas.style.height = `${event.target.height}px`;
  parent.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  const drawingUtils = new DrawingUtils(ctx);

  // İki tespiti birden yap
  const poseResult = await poseLandmarker.detect(event.target);
  const handResult = await handLandmarker.detect(event.target);

  // Çizimleri yap
  poseResult.landmarks.forEach(landmark => {
    drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
    drawingUtils.drawLandmarks(landmark, { radius: 2 });
  });

  handResult.landmarks.forEach(landmark => {
    drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS);
    drawingUtils.drawLandmarks(landmark, { color: "#FF0000", lineWidth: 1 });
  });
}

// --- Canlı Kamera Takibi ---
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const webcamButton = document.getElementById("webcamButton");
webcamButton.addEventListener("click", () => {
  if (webcamRunning) {
    webcamRunning = false;
    webcamButton.innerText = "KAMERAYI ETKİNLEŞTİR";
  } else {
    webcamRunning = true;
    webcamButton.innerText = "KAMERAYI KAPAT";
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    });
  }
});

let lastVideoTime = -1;
async function predictWebcam() {
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    // Senkronize tespit
    const poseResult = poseLandmarker.detectForVideo(video, startTimeMs);
    const handResult = handLandmarker.detectForVideo(video, startTimeMs);

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Vücut Çizimi
    if (poseResult.landmarks) {
      for (const landmark of poseResult.landmarks) {
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        drawingUtils.drawLandmarks(landmark, { radius: 2 });
      }
    }

    // El Çizimi
    if (handResult.landmarks) {
      for (const landmark of handResult.landmarks) {
        drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS);
        drawingUtils.drawLandmarks(landmark, { color: "#FF0000", lineWidth: 2 });
      }
    }
    canvasCtx.restore();
  }

  if (webcamRunning) window.requestAnimationFrame(predictWebcam);
}
