/* Emoji Face Detector — face-api.js (client-only)
 * - Detects faces + expressions, overlays emoji per face
 * - Works with webcam or uploaded image
 * - GitHub Pages friendly
 */

// ====== DOM ======
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const btnCamera = document.getElementById('btnCamera');
const btnStop = document.getElementById('btnStop');
const btnShot = document.getElementById('btnShot');
const mirrorChk = document.getElementById('mirror');
const boxesChk = document.getElementById('showBoxes');
const sizeRange = document.getElementById('emojiSize');
const fileInput = document.getElementById('fileInput');
const statusEl = document.getElementById('status');
const exprList = document.getElementById('exprList');
const loading = document.getElementById('loading');

// ====== Config ======
const MODEL_URL = './models'; // put weights here (see README below)
const TFD_OPTS = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

// Map expression -> emoji
const EMOJI = {
  neutral: '🙂', happy: '😄', sad: '😢', angry: '😠',
  fearful: '😱', disgusted: '🤢', surprised: '😮', default: '🙂'
};

let stream = null;
let running = false;
let lastTime = performance.now();

// ====== Helpers ======
function setStatus(t){ statusEl.textContent = t; }
function bestExpression(expressions){
  // returns [name, score]
  let maxName = 'neutral', maxScore = 0;
  for(const [name,score] of Object.entries(expressions)){
    if(score > maxScore){ maxScore = score; maxName = name; }
  }
  return [maxName, maxScore];
}
function drawFrame(source){
  // Draw current frame (video or image) to canvas, optionally mirrored
  const W = canvas.width, H = canvas.height;
  ctx.save();
  ctx.clearRect(0,0,W,H);
  if(mirrorChk.checked){
    ctx.translate(W,0); ctx.scale(-1,1);
  }
  ctx.drawImage(source, 0, 0, W, H);
  ctx.restore();
}
function drawBoxesAndEmojis(detections){
  const W = canvas.width, H = canvas.height;
  const emojiSize = Number(sizeRange.value);

  // Build overall top-3 expressions for sidebar
  const counts = {};
  detections.forEach(det=>{
    const [name] = bestExpression(det.expressions);
    counts[name] = (counts[name]||0)+1;
  });
  exprList.innerHTML = Object.entries(counts)
    .sort((a,b)=>b[1]-a[1])
    .slice(0,5)
    .map(([n,c])=>`<li>${EMOJI[n]||EMOJI.default} ${n} — ${c}</li>`).join('');

  ctx.save();
  if(mirrorChk.checked){ ctx.translate(W,0); ctx.scale(-1,1); }

  for(const det of detections){
    const box = det.detection.box; // {x,y,width,height}
    const [name, score] = bestExpression(det.expressions);
    const emoji = EMOJI[name] || EMOJI.default;

    // optional: box
    if(boxesChk.checked){
      ctx.strokeStyle = 'rgba(124,197,255,.9)';
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      ctx.fillStyle = 'rgba(0,0,0,.6)';
      ctx.fillRect(box.x, Math.max(0, box.y-18), 120, 18);
      ctx.fillStyle = '#e6edf3';
      ctx.font = '12px ui-sans-serif,system-ui';
      ctx.fillText(`${name} ${(score*100|0)}%`, box.x+4, Math.max(12, box.y-6));
    }

    // emoji size = slider, centered above the face box
    const size = Math.min(emojiSize, Math.max(32, box.width*0.7));
    ctx.font = `${size}px Apple Color Emoji, Segoe UI Emoji, Noto Color Emoji, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    const ex = box.x + box.width/2;
    const ey = box.y; // top edge of face box
    ctx.fillText(emoji, ex, Math.max(10, ey-6));
  }

  ctx.restore();
}

// ====== Camera ======
async function startCamera(){
  if(running) return;
  try{
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio:false });
    video.srcObject = stream; await video.play();
    // Fit canvas to video aspect
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 540;
    running = true;
    btnCamera.disabled = true; btnStop.disabled = false;
    loop();
  }catch(err){
    console.error(err); setStatus('Camera error: ' + err.message);
  }
}
function stopCamera(){
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  running = false; btnCamera.disabled = false; btnStop.disabled = true;
}

// ====== Main detect loop ======
async function loop(){
  if(!running) return;
  const now = performance.now();
  const dt = now - lastTime; lastTime = now;

  // Draw current frame
  drawFrame(video);

  // Detect faces + expressions
  const detections = await faceapi
    .detectAllFaces(canvas, TFD_OPTS)
    .withFaceExpressions();

  // Overlay
  drawBoxesAndEmojis(detections);

  setStatus(`Faces: ${detections.length} • ${(1000/dt|0)} fps`);
  requestAnimationFrame(loop);
}

// ====== Image file mode ======
fileInput.addEventListener('change', async e=>{
  const file = e.target.files?.[0]; if(!file) return;
  stopCamera();
  const img = new Image();
  img.onload = async ()=>{
    // Fit canvas to image
    const maxW = 1280;
    const scale = Math.min(1, maxW / img.width);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    drawFrame(img);
    const detections = await faceapi
      .detectAllFaces(canvas, TFD_OPTS)
      .withFaceExpressions();
    drawBoxesAndEmojis(detections);
    setStatus(`Image • Faces: ${detections.length}`);
  };
  img.src = URL.createObjectURL(file);
});

// ====== Snapshot ======
btnShot.addEventListener('click', ()=>{
  const a = document.createElement('a');
  a.download = `emoji-face-${Date.now()}.png`;
  a.href = canvas.toDataURL('image/png'); a.click();
});

// ====== UI wiring ======
btnCamera.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);
[sizeRange, boxesChk, mirrorChk].forEach(el=>{
  el.addEventListener('input', ()=>{
    // redraw current frame (video or last image) with updated settings
    if(running) { /* next loop will redraw */ }
  });
});

// ====== Bootstrap: load models once ======
(async function boot(){
  // Show loading badge until all model files are fetched
  loading.style.display = 'block';
  setStatus('Loading models… (first time takes a few seconds)');
  // We need: tiny face detector + face expression classifier
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
  loading.style.display = 'none';
  setStatus('Ready — choose Camera or load an Image');
})();
