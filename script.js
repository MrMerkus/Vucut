import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

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
      delegate: "GPU"
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

  document.body.classList.add("ready");
};
setupModels();

const webcamButton = document.getElementById("webcamButton");
webcamButton.addEventListener("click", async () => {
  if (webcamRunning) {
    webcamRunning = false;
    webcamButton.innerText = "KAMERAYI AÇ";
    video.srcObject.getTracks().forEach(t => t.stop());
  } else {
    webcamRunning = true;
    webcamButton.innerText = "KAMERAYI KAPAT";
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  }
});

let lastVideoTime = -1;
async function predictWebcam() {
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
