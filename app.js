/* Emoji Face Detector — upgraded features
 * - Bounding boxes (toggle)
 * - 68 landmarks (toggle)
 * - Face blur/mosaic (toggle)
 * - Multi-face tracking IDs + trails
 * - Expression smoothing (EMA)
 * - Detector tuning: inputSize + threshold
 * - HiDPI canvas scaling
 * - Download detections JSON
 */

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const btnCamera = document.getElementById('btnCamera');
const btnStop = document.getElementById('btnStop');
const btnShot = document.getElementById('btnShot');
const btnDump = document.getElementById('btnDump');
const mirrorChk = document.getElementById('mirror');
const boxesChk = document.getElementById('showBoxes');
const lmChk = document.getElementById('showLM');
const blurChk = document.getElementById('blurFaces');
const trailsChk = document.getElementById('showTrails');
const sizeRange = document.getElementById('emojiSize');
const inpSizeRange = document.getElementById('inpSize');
const thrRange = document.getElementById('thr');
const fileInput = document.getElementById('fileInput');
const statusEl = document.getElementById('status');
const exprList = document.getElementById('exprList');
const loading = document.getElementById('loading');

// ====== paths & options ======
const MODEL_BASE = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights'; // robust on GH Pages

function TFD() {
  return new faceapi.TinyFaceDetectorOptions({
    inputSize: Number(inpSizeRange.value),
    scoreThreshold: Number(thrRange.value)
  });
}

const EMOJI = {
  neutral: '🙂', happy: '😄', sad: '😢', angry: '😠',
  fearful: '😱', disgusted: '🤢', surprised: '😮', default: '🙂'
};

// ====== runtime ======
let stream = null;
let running = false;
let lastFrameTime = performance.now();
let lastDetections = [];  // for JSON dump
const trails = [];        // [{x,y,life,id}]
const tracks = new Map(); // id -> {cx,cy,lastSeen,exprEMA:{...}}
let nextId = 1;

// ====== utils ======
const clamp = (v,a,b)=>Math.max(a,Math.min(b,v));
const setStatus = (t)=> statusEl.textContent = t;

function fitCanvasToSource(w, h) {
  // HiDPI for crisp drawing
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawFrame(source) {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  ctx.save();
  ctx.clearRect(0,0,W,H);
  if (mirrorChk.checked) { ctx.translate(W,0); ctx.scale(-1,1); }
  ctx.drawImage(source, 0, 0, W, H);
  ctx.restore();
}

function centerOf(box) {
  return { x: box.x + box.width/2, y: box.y + box.height/2 };
}

// simple nearest-neighbor tracker
function assignTracks(detections) {
  const now = performance.now();
  const used = new Set();
  for (const det of detections) {
    const c = centerOf(det.detection.box);
    let bestId = null, bestD = 99999;
    // find nearest active track
    for (const [id, t] of tracks.entries()) {
      if (now - t.lastSeen > 1500) continue; // stale
      const d = Math.hypot(c.x - t.cx, c.y - t.cy);
      if (d < bestD) { bestD = d; bestId = id; }
    }
    if (bestId != null && bestD < Math.max(det.detection.box.width, det.detection.box.height)) {
      // update existing
      const t = tracks.get(bestId);
      t.cx = c.x; t.cy = c.y; t.lastSeen = now;
      // EMA expressions
      t.exprEMA = emaExpressions(t.exprEMA, det.expressions, 0.6);
      det.trackId = bestId;
    } else {
      // new track
      const id = nextId++;
      tracks.set(id, { cx: c.x, cy: c.y, lastSeen: now, exprEMA: det.expressions });
      det.trackId = id;
    }
  }
  // cleanup stale
  for (const [id, t] of tracks.entries()) {
    if (now - t.lastSeen > 3000) tracks.delete(id);
  }
}

function emaExpressions(prev, cur, alpha=0.6) {
  if (!prev) return cur;
  const out = {};
  for (const k of Object.keys(cur)) out[k] = alpha*prev[k] + (1-alpha)*cur[k];
  return out;
}

function bestExpression(exprs) {
  let name='neutral', score=0;
  for (const [k,v] of Object.entries(exprs)) if (v > score) { score=v; name=k; }
  return [name, score];
}

function drawBoxesAndEmojis(detections) {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const emojiSize = Number(sizeRange.value);

  // sidebar summary
  const counts = {};
  detections.forEach(det=>{
    const em = det.trackId && tracks.get(det.trackId)?.exprEMA || det.expressions;
    const [name] = bestExpression(em);
    counts[name] = (counts[name]||0) + 1;
  });
  exprList.innerHTML = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,5)
    .map(([n,c])=>`<li>${EMOJI[n]||EMOJI.default} ${n} — ${c}</li>`).join('');

  ctx.save();
  if (mirrorChk.checked) { ctx.translate(W,0); ctx.scale(-1,1); }

  for (const det of detections) {
    const box = det.detection.box;
    const id = det.trackId ?? '?';
    const em = id && tracks.get(id)?.exprEMA || det.expressions;
    const [name, score] = bestExpression(em);
    const emoji = EMOJI[name] || EMOJI.default;

    // blur face region if requested
    if (blurChk.checked) {
      mosaicRegion(box, 8);
    }

    // draw bbox
    if (boxesChk.checked) {
      ctx.strokeStyle = 'rgba(124,197,255,.95)';
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // label
      ctx.fillStyle = 'rgba(0,0,0,.6)';
      ctx.fillRect(box.x, Math.max(0, box.y-22), Math.max(120, 48), 20);
      ctx.fillStyle = '#e6edf3';
      ctx.font = '12px ui-sans-serif,system-ui';
      ctx.fillText(`ID ${id} • ${name} ${(score*100|0)}%`, box.x+6, Math.max(12, box.y-6));
    }

    // landmarks
    if (lmChk.checked && det.landmarks) {
      const pts = det.landmarks.positions;
      ctx.fillStyle = '#7cc5ff';
      for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 1.8, 0, Math.PI*2); ctx.fill(); }
    }

    // emoji above face
    const size = Math.min(emojiSize, Math.max(32, box.width*0.7));
    ctx.font = `${size}px Apple Color Emoji, Segoe UI Emoji, Noto Color Emoji, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(emoji, box.x + box.width/2, Math.max(10, box.y-6));

    // trail dot at center
    if (trailsChk.checked) {
      const c = centerOf(box);
      trails.push({ x:c.x, y:c.y, life:30, id });
    }
  }
  ctx.restore();

  // draw trails (not mirrored to keep consistent look)
  if (trailsChk.checked) {
    for (let i=trails.length-1;i>=0;i--) {
      const t = trails[i];
      ctx.fillStyle = `rgba(124,197,255,${t.life/30 * 0.8})`;
      ctx.beginPath(); ctx.arc(t.x, t.y, 3, 0, Math.PI*2); ctx.fill();
      t.life--; if (t.life <= 0) trails.splice(i,1);
    }
  }
}

function mosaicRegion(box, factor=8) {
  // pixelate a rectangular region
  const sx = Math.max(0, box.x|0), sy = Math.max(0, box.y|0);
  const sw = Math.max(1, box.width|0), sh = Math.max(1, box.height|0);
  const smallW = Math.max(1, (sw/factor)|0), smallH = Math.max(1, (sh/factor)|0);

  const tmp = document.createElement('canvas');
  tmp.width = smallW; tmp.height = smallH;
  const tctx = tmp.getContext('2d');
  tctx.imageSmoothingEnabled = false;

  // draw region scaled down then back up to create mosaic
  tctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, smallW, smallH);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, smallW, smallH, sx, sy, sw, sh);
  ctx.imageSmoothingEnabled = true;
}

// ====== camera ======
async function startCamera(){
  if (running) return;
  try{
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio:false });
    video.srcObject = stream; await video.play();
    fitCanvasToSource(video.videoWidth || 960, video.videoHeight || 540);
    running = true;
    btnCamera.disabled = true; btnStop.disabled = false;
    loop();
  }catch(err){
    console.error(err); setStatus('Camera error: ' + err.message);
  }
}
function stopCamera(){
  if (stream) { stream.getTracks().forEach(t=>t.stop()); stream=null; }
  running = false; btnCamera.disabled = false; btnStop.disabled = true;
}

// ====== detection loop ======
async function loop(){
  if (!running) return;

  // 1) draw current video frame to canvas
  drawFrame(video);

  // 2) detect (with or without landmarks)
  const opts = new faceapi.TinyFaceDetectorOptions({
    inputSize: Number(inpSizeRange.value),
    scoreThreshold: Number(thrRange.value)
  });

  const detections = lmChk.checked
    ? await faceapi.detectAllFaces(canvas, opts).withFaceLandmarks().withFaceExpressions()
    : await faceapi.detectAllFaces(canvas, opts).withFaceExpressions();

  // 3) tracking + smoothing
  assignTracks(detections);

  // 4) overlays in real time (boxes, landmarks, blur, trails, emoji)
  drawBoxesAndEmojis(detections);

  requestAnimationFrame(loop);
}

function serializeDetections(ds) {
  const arr = ds.map(d => ({
    id: d.trackId ?? null,
    box: { x:+d.detection.box.x.toFixed(1), y:+d.detection.box.y.toFixed(1),
           w:+d.detection.box.width.toFixed(1), h:+d.detection.box.height.toFixed(1) },
    expressions: d.expressions
  }));
  return { ts: Date.now(), count: arr.length, detections: arr };
}

// ====== image mode ======
fileInput.addEventListener('change', async e=>{
  const file = e.target.files?.[0]; if (!file) return;
  stopCamera();
  const img = new Image();
  img.onload = async ()=>{
    const maxW = 1280;
    const scale = Math.min(1, maxW / img.width);
    fitCanvasToSource(Math.round(img.width*scale), Math.round(img.height*scale));
    drawFrame(img);

    let detections;
    if (lmChk.checked) {
      detections = await faceapi.detectAllFaces(canvas, TFD()).withFaceLandmarks().withFaceExpressions();
    } else {
      detections = await faceapi.detectAllFaces(canvas, TFD()).withFaceExpressions();
    }
    assignTracks(detections);
    drawBoxesAndEmojis(detections);
    lastDetections = serializeDetections(detections);
    setStatus(`Image • Faces: ${detections.length}`);
  };
  img.src = URL.createObjectURL(file);
});

// ====== snapshot & dump ======
btnShot.addEventListener('click', ()=>{
  const a = document.createElement('a');
  a.download = `emoji-face-${Date.now()}.png`;
  a.href = canvas.toDataURL('image/png'); a.click();
});
btnDump.addEventListener('click', ()=>{
  const blob = new Blob([JSON.stringify(lastDetections, null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.download = `detections-${Date.now()}.json`;
  a.href = URL.createObjectURL(blob); a.click();
});

// ====== UI wiring ======
btnCamera.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);
[sizeRange, boxesChk, mirrorChk, lmChk, blurChk, trailsChk].forEach(el=>{
  el.addEventListener('input', ()=>{/* next frame will use new settings */});
});
[inpSizeRange, thrRange].forEach(el=>{
  el.addEventListener('input', ()=>{
    // briefly show new settings in status
    setStatus(`inputSize ${inpSizeRange.value} • thr ${thrRange.value}`);
  });
});

// ====== boot (load models) ======
(async function boot(){
  loading.style.display = 'block';
  setStatus('Loading models… (first time takes a few seconds)');

  // detector + expressions (required)
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_BASE);
  await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_BASE);
  // landmarks (optional toggle)
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_BASE);

  loading.style.display = 'none';
  setStatus('Ready — choose Camera or load an Image');
})();
