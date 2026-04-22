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

// Görünürlük Eşiği (0.0 ile 1.0 arası). 
// 0.65 ve üzeri demek: "Kameranın net bir şekilde görmediği vücut kısımlarını çizme" demektir.
const VISIBILITY_THRESHOLD = 0.65; 

// Modelleri yükle (Tam GPU ve Full Model Odaklı)
const setupModels = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  
  // VÜCUT TAKİBİ - Titreme azaltıldı, model büyütüldü
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      // 'lite' yerine 'full' kullanıyoruz. İşlem yükü artar ama GPU bunu çözer. Titreme biter.
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
      delegate: "GPU" // İşlemleri Ekran Kartına Yönlendir
    },
    runningMode: "VIDEO",
    numPoses: 2,
    minPoseDetectionConfidence: 0.75, // Kararlılık için minimum algılama güveni artırıldı
    minPosePresenceConfidence: 0.75,
    minTrackingConfidence: 0.75      // Titremeyi engelleyen en önemli parametre
  });

  // EL TAKİBİ
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU" // İşlemleri Ekran Kartına Yönlendir
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
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 1280, height: 720 } // Yüksek çözünürlük isteği
    });
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  }
});

let lastVideoTime = -1;
async function predictWebcam() {
  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  let startTimeMs = performance.now();
  
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // 1. VÜCUT TAHMİNİ (Özel Görünürlük Filtreli Çizim)
    const poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
    if (poseResults.landmarks) {
      for (const pose of poseResults.landmarks) {
        
        // Önce Bağlantıları (Çizgileri) Çiz
        PoseLandmarker.POSE_CONNECTIONS.forEach(connection => {
          const startPoint = pose[connection.start];
          const endPoint = pose[connection.end];

          // Sadece her iki nokta da kamerada net görünüyorsa çizgiyi çek
          if (startPoint.visibility > VISIBILITY_THRESHOLD && endPoint.visibility > VISIBILITY_THRESHOLD) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(startPoint.x * canvasElement.width, startPoint.y * canvasElement.height);
            canvasCtx.lineTo(endPoint.x * canvasElement.width, endPoint.y * canvasElement.height);
            canvasCtx.strokeStyle = "#00bfa5";
            canvasCtx.lineWidth = 3;
            canvasCtx.stroke();
          }
        });

        // Sonra Noktaları (Eklemleri) Çiz
        pose.forEach(landmark => {
          // Eğer eklem kameranın görmediği bir yerdeyse (masanın altı vb.) noktayı koyma
          if (landmark.visibility > VISIBILITY_THRESHOLD) {
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
        // Eller genelde bütün olarak göründüğü için standart çizim aracı yeterlidir
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
