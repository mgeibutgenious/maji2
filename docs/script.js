// ===== Model / Labels / Sizes =====
let model;
const classLabels = ['Big Lot', 'C Press', 'Snyders'];
const INPUT_SIZE = 224;

let videoEl = null;
let running = false;

// Freeze state after capture
let isFrozen = false;        // UI/loop freeze flag
let frozenDataURL = null;    // to show frozen still in <video> poster if needed

// ===== EXACT model math you used =====
function makeInputFromVideo(video) {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(video)
      .resizeNearestNeighbor([INPUT_SIZE, INPUT_SIZE])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims(); // [1,H,W,3]
    return tensor;
  });
}

async function loadModel() {
  model = await tf.loadLayersModel('tfjs_model/model.json');
  tf.tidy(() => {
    const z = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    const y = model.predict(z);
    if (Array.isArray(y)) y.forEach(t => t.dispose()); else y.dispose?.();
  });
  console.log("Model loaded");
}

async function setupCamera() {
  const el = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" }, audio: false
  });
  el.srcObject = stream;
  await new Promise(res => (el.onloadedmetadata = () => res()));
  await el.play();
  videoEl = el;
  return el;
}

// ===== UI helpers =====
function renderBoth(predArray) {
  // Original block
  const resultP = document.getElementById('result');
  let text = '';
  for (let i = 0; i < classLabels.length; i++) {
    text += `${classLabels[i]}: ${(predArray[i] * 100).toFixed(2)}%\n`;
  }
  resultP.innerText = text.trimEnd();

  // Vertical list with best highlighted
  const items = classLabels.map((label, i) => ({ label, p: predArray[i] ?? 0 }));
  let bestIdx = 0, bestVal = -Infinity;
  for (let i = 0; i < items.length; i++) if (items[i].p > bestVal) { bestVal = items[i].p; bestIdx = i; }

  const predsEl = document.getElementById('predictions');
  predsEl.innerHTML = items.map((it, i) => `
    <div class="row ${i === bestIdx ? 'best' : ''}">
      <div>${it.label}</div>
      <div>${(it.p * 100).toFixed(1)}%</div>
    </div>
  `).join('');

  // store latest top for capture naming
  lastTop = { label: items[bestIdx].label, confidence: items[bestIdx].p };
}

async function selectBackend() {
  try { await tf.setBackend('webgl'); } catch (_) {}
  if (tf.getBackend() !== 'webgl') {
    try { await tf.setBackend('wasm'); } catch (_) {}
  }
  await tf.ready();
  const b = document.getElementById('backend');
  if (b) b.textContent = `backend: ${tf.getBackend()}`;
}

// ===== Predict Loop =====
let lastTop = null;

async function predictLoop() {
  if (!running) return;
  if (isFrozen) return; // freeze: do not update percentages

  await tf.nextFrame();
  try {
    const input = makeInputFromVideo(videoEl);
    const prediction = model.predict(input);
    const predArray = await prediction.data();
    renderBoth(predArray);
    tf.dispose([input, prediction]);
  } catch (e) {
    console.error(e);
    document.getElementById('err').textContent = String(e?.message || e);
  }
  requestAnimationFrame(predictLoop);
}

// ===== Start & Cleanup =====
function stopStreamAndFreezePoster(blob) {
  // Stop tracks, keep a still frame visible
  if (videoEl) {
    // assign still image to video poster via dataURL
    const url = URL.createObjectURL(blob);
    frozenDataURL = url;
    // Some browsers show last frame if paused; ensure with poster snapshot using <img> overlay technique:
    // Simpler: pause and keep video element (last frame visible in most browsers)
    videoEl.pause();

    if (videoEl.srcObject) {
      for (const track of videoEl.srcObject.getVideoTracks()) track.stop();
      videoEl.srcObject = null;
    }
  }
}

function clearFrozenPoster() {
  if (frozenDataURL) {
    URL.revokeObjectURL(frozenDataURL);
    frozenDataURL = null;
  }
}

function stop() {
  running = false;
  const s = document.getElementById('status'); if (s) s.textContent = '停止中';
}

function disposeAll() {
  stop();
  if (model) { model.dispose(); model = undefined; }
  if (videoEl && videoEl.srcObject) {
    for (const track of videoEl.srcObject.getVideoTracks()) track.stop();
    videoEl.srcObject = null;
  }
  tf.engine().disposeVariables();
  tf.backend().dispose?.();
}

// ===== Local saving with real folders (File System Access API) =====
let baseDirHandle = null;

async function chooseBaseFolder() {
  if (!window.showDirectoryPicker) {
    document.getElementById('saveStatus').textContent =
      'ブラウザがフォルダ保存に未対応です（Chrome/Edge 推奨）。ダウンロード保存に切替えます。';
    return;
  }
  baseDirHandle = await window.showDirectoryPicker();
  document.getElementById('saveStatus').textContent =
    '保存先を設定しました。ここに自動保存します。';
}

async function ensureSubdir(parent, name) {
  return await parent.getDirectoryHandle(name, { create: true });
}

function yyyymmdd() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth()+1).padStart(2,'0');
  const dd = String(d.getDate()).padStart(2,'0');
  return `${yyyy}${mm}${dd}`;
}

function captureFrameToBlob() {
  return new Promise((resolve) => {
    const canvas = document.getElementById('captureCanvas');
    const w = videoEl.videoWidth, h = videoEl.videoHeight;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, w, h);
    canvas.toBlob((blob) => resolve(blob), 'image/png', 0.92);
  });
}

async function writeBlobTo(handle, filename, blob) {
  const fileHandle = await handle.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(blob);
  await writable.close();
}

function downloadFallback(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function saveCaptured(whereFolder, folderClass, baseLabel, conf, blob) {
  const date = yyyymmdd();
  const confPct = (conf * 100).toFixed(0);
  const filename = `${date}_${baseLabel}_${confPct}%.png`;

  try {
    if (baseDirHandle) {
      const root = await ensureSubdir(baseDirHandle, whereFolder);
      const classDir = await ensureSubdir(root, folderClass);
      await writeBlobTo(classDir, filename, blob);
      document.getElementById('saveStatus').textContent =
        `保存しました: ${whereFolder}/${folderClass}/${filename}`;
    } else {
      downloadFallback(`${whereFolder}_${folderClass}_${filename}`, blob);
      document.getElementById('saveStatus').textContent =
        `ダウンロードしました: ${whereFolder}/${folderClass}/${filename}`;
    }
  } catch (e) {
    console.error(e);
    document.getElementById('saveStatus').textContent = `保存エラー: ${e.message || e}`;
  } finally {
    // hide choice buttons after save
    elAgree.style.display = 'none';
    elDisagree.style.display = 'none';
    elChooseBig.style.display = 'none';
    elChooseC.style.display = 'none';
    elChooseS.style.display = 'none';
  }
}

// ===== Capture + Agree/Disagree flow =====
let capturedBlob = null;
let cameraJudgment = null;
let cameraConfidence = null;

const elAgree = document.getElementById('agreeBtn');
const elDisagree = document.getElementById('disagreeBtn');
const elCap = document.getElementById('captureBtn');
const elChooseBig = document.getElementById('chooseBigLot');
const elChooseC = document.getElementById('chooseCPress');
const elChooseS = document.getElementById('chooseSnyders');

elCap.addEventListener('click', async () => {
  document.getElementById('saveStatus').textContent = '';
  if (!videoEl || !lastTop) {
    document.getElementById('saveStatus').textContent =
      'まだ予測がありません。少し待ってから再試行してください。';
    return;
  }

  // 1) capture current frame
  capturedBlob = await captureFrameToBlob();
  cameraJudgment = lastTop.label;
  cameraConfidence = lastTop.confidence;

  // 2) freeze UI: stop loop, pause & release camera so it feels like a photo
  isFrozen = true;
  running = false;
  stopStreamAndFreezePoster(capturedBlob);
  document.getElementById('status').textContent = 'キャプチャ中（停止）';

  // 3) show Agree/Disagree
  elAgree.style.display = '';
  elAgree.textContent = 'Agree';         // ensure text shown
  elDisagree.style.display = '';
  elDisagree.textContent = 'Disagree';   // ensure text shown

  // hide class choices until Disagree
  elChooseBig.style.display = 'none';
  elChooseC.style.display = 'none';
  elChooseS.style.display = 'none';
});

elAgree.addEventListener('click', async () => {
  if (!capturedBlob) return;
  await saveCaptured('Agree', cameraJudgment, cameraJudgment, cameraConfidence, capturedBlob);
});

elDisagree.addEventListener('click', () => {
  if (!capturedBlob) return;
  elChooseBig.style.display = '';
  elChooseC.style.display = '';
  elChooseS.style.display = '';
});

elChooseBig.addEventListener('click', async () => {
  if (!capturedBlob) return;
  await saveCaptured('Disagree', 'Big Lot', cameraJudgment, cameraConfidence, capturedBlob);
});
elChooseC.addEventListener('click', async () => {
  if (!capturedBlob) return;
  await saveCaptured('Disagree', 'C Press', cameraJudgment, cameraConfidence, capturedBlob);
});
elChooseS.addEventListener('click', async () => {
  if (!capturedBlob) return;
  await saveCaptured('Disagree', 'Snyders', cameraJudgment, cameraConfidence, capturedBlob);
});

// ===== Buttons & Init =====
document.getElementById('chooseFolderBtn').addEventListener('click', chooseBaseFolder);

document.getElementById('startBtn').addEventListener('click', async () => {
  document.getElementById('err').textContent = '';

  // if we were frozen after capture, resume: re-open camera
  if (isFrozen) {
    clearFrozenPoster();
    isFrozen = false;
  }

  if (!videoEl || !videoEl.srcObject) {
    await setupCamera();
  }
  if (!model) await loadModel();

  running = true;
  const s = document.getElementById('status'); if (s) s.textContent = '推論中…';
  requestAnimationFrame(predictLoop);
});

window.addEventListener('DOMContentLoaded', async () => {
  try {
    await selectBackend();
    await setupCamera();
    await loadModel();
    running = true;
    const s = document.getElementById('status'); if (s) s.textContent = '推論中…';
    requestAnimationFrame(predictLoop);
  } catch (e) {
    console.error(e);
    document.getElementById('err').textContent = '初期化エラー: ' + (e?.message || e);
  }
});

window.addEventListener('pagehide', disposeAll);
window.addEventListener('beforeunload', disposeAll);
