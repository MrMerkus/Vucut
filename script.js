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
  
  // 1. TİTREME ÇÖZÜMÜ: En hassas model olan "HEAVY" modeline geçildi.
  // Ek olarak algılama güvenliği (Confidence) eşikleri artırıldı.
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    minPoseDetectionConfidence: 0.65, // %65 emin olmadan algılama
    minPosePresenceConfidence: 0.65,
    minTrackingConfidence: 0.65       // Takipte hassasiyeti artır
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

  document.body.classList.add("ready");
};
setupModels();

const webcamButton = document.getElementById("webcamButton");
webcamButton.addEventListener("click", async () => {
  if (!poseLandmarker || !handLandmarker) {
    alert("Yapay zeka modelleri henüz indiriliyor, lütfen birkaç saniye daha bekleyin.");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    webcamButton.innerText = "KAMERAYI AÇ";
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  } else {
    webcamRunning = true;
    webcamButton.innerText = "KAMERAYI KAPAT";
    
    // 2. SAPMA ÇÖZÜMÜ: Kamerayı CSS'teki 16:9 oranıyla eşleşmesi için zorla 720p'de başlat
    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user" // Ön/Web kamerasını tercih et
      }
    };
    
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    video.onloadeddata = async () => {
      await video.play();
      predictWebcam();
    };
  }
});

let lastVideoTime = -1;
async function predictWebcam() {
  if (!webcamRunning) return;

  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  let startTimeMs = performance.now();
  
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    const poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
    const handResults = handLandmarker.detectForVideo(video, startTimeMs);

    // Vücut için Turkuaz Neon Efekti
    if (poseResults.landmarks) {
      for (const landmark of poseResults.landmarks) {
        canvasCtx.shadowColor = "#00bfa5";
        canvasCtx.shadowBlur = 15;
        
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { 
          color: "#00bfa5", 
          lineWidth: 4 
        });
        drawingUtils.drawLandmarks(landmark, { 
          color: "#ffffff", 
          fillColor: "#00bfa5",
          lineWidth: 2,
          radius: 5 
        });
      }
    }

    // Eller için Pembe Neon Efekti
    if (handResults.landmarks) {
      for (const landmarks of handResults.landmarks) {
        canvasCtx.shadowColor = "#ff4081";
        canvasCtx.shadowBlur = 15;

        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { 
          color: "#ff4081", 
          lineWidth: 3 
        });
        drawingUtils.drawLandmarks(landmarks, { 
          color: "#ffffff", 
          fillColor: "#ff4081",
          lineWidth: 2,
          radius: 4 
        });
      }
    }
    
    canvasCtx.restore();
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}
