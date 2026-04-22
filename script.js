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

// GÖRÜNÜRLÜK FİLTRESİ ÇOK DAHA KATI HALE GETİRİLDİ (0.85)
// Sadece net olarak kadrajda olan kısımlar çizilecek.
const VISIBILITY_THRESHOLD = 0.85; 

const setupModels = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 2,
    // Kararlılık için güven oranları %85'e çıkarıldı
    minPoseDetectionConfidence: 0.85, 
    minPosePresenceConfidence: 0.85,
    minTrackingConfidence: 0.85      
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.7,
    minTrackingConfidence: 0.7
  });

  document.querySelector('.status').innerText = "GPU Sistem Aktif. Kamerayı açabilirsiniz.";
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
    // İdeal bir çözünürlük isteyelim ama kameranın orijinaline saygı duyalım
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 1280, height: 720 } 
    });
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  }
});

let lastVideoTime = -1;
async function predictWebcam() {
  // KAYMA SORUNUNU ÇÖZEN EN KRİTİK KISIM:
  // Canvas'ın hem iç çözünürlüğünü hem de CSS görünüm boyutunu videoya zorla eşitliyoruz.
  if (canvasElement.width !== video.videoWidth || canvasElement.height !== video.videoHeight) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    canvasElement.style.width = video.videoWidth + "px";
    canvasElement.style.height = video.videoHeight + "px";
  }

  let startTimeMs = performance.now();
  
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // 1. VÜCUT TAHMİNİ
    const poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
    if (poseResults.landmarks) {
      for (const pose of poseResults.landmarks) {
        
        // Çizgiler (Bağlantılar)
        PoseLandmarker.POSE_CONNECTIONS.forEach(connection => {
          const startPoint = pose[connection.start];
          const endPoint = pose[connection.end];

          // Her iki nokta da net görünüyorsa ve ekranda (x/y 0 ile 1 arasında) ise çiz
          if (startPoint.visibility > VISIBILITY_THRESHOLD && endPoint.visibility > VISIBILITY_THRESHOLD &&
              startPoint.x >= 0 && startPoint.x <= 1 && endPoint.x >= 0 && endPoint.x <= 1) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(startPoint.x * canvasElement.width, startPoint.y * canvasElement.height);
            canvasCtx.lineTo(endPoint.x * canvasElement.width, endPoint.y * canvasElement.height);
            canvasCtx.strokeStyle = "#00bfa5";
            canvasCtx.lineWidth = 3;
            canvasCtx.stroke();
          }
        });

        // Noktalar (Eklemler)
        pose.forEach(landmark => {
          if (landmark.visibility > VISIBILITY_THRESHOLD && landmark.x >= 0 && landmark.x <= 1 && landmark.y >= 0 && landmark.y <= 1) {
            canvasCtx.beginPath();
            canvasCtx.arc(landmark.x * canvasElement.width, landmark.y * canvasElement.height, 4, 0, 2 * Math.PI);
            canvasCtx.fillStyle = "#ffffff";
            canvasCtx.fill();
          }
        });
      }
    }

    // 2. EL TAHMİNİ
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
