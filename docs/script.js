let model;
const classLabels = ['Big Lot', 'C Press', 'Snyders'];
const INPUT_SIZE = 224;

let videoEl = null;
let freezeImg = null;
let running = false;
let isFrozen = false;

let lastTop = null;
let baseDirHandle = null;

function makeInputFromVideo(video){
  return tf.tidy(()=> tf.browser.fromPixels(video)
    .resizeNearestNeighbor([INPUT_SIZE, INPUT_SIZE])
    .toFloat()
    .div(tf.scalar(255))
    .expandDims());
}

async function loadModel(){
  model = await tf.loadLayersModel('tfjs_model/model.json');
  tf.tidy(()=>{
    const z = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    const y = model.predict(z);
    (Array.isArray(y)? y.forEach(t=>t.dispose()): y.dispose?.());
  });
}

async function setupCamera(){
  const el = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:"environment" }, audio:false });
  el.srcObject = stream;
  await new Promise(r=> el.onloadedmetadata = ()=> r());
  await el.play();
  videoEl = el;
  freezeImg = document.getElementById('freezeOverlay');
  return el;
}

/* ===== render list with percentages only (top highlighted) ===== */
function renderListWithPerc(probs, bestIdx){
  const predsEl = document.getElementById('predictions');
  predsEl.innerHTML = classLabels.map((label,i)=>`
    <div class="row ${i===bestIdx?'best':''}">
      <div>${label}</div>
      <div>${(probs[i]*100).toFixed(1)}%</div>
    </div>`).join('');
}

/* ===== loop ===== */
async function predictLoop(){
  if(!running || isFrozen) return;
  await tf.nextFrame();
  try{
    const input = makeInputFromVideo(videoEl);
    const out = model.predict(input);
    const probs = await out.data();

    // best idx
    let bestIdx=0, best=probs[0];
    for(let i=1;i<probs.length;i++) if(probs[i]>best){ best=probs[i]; bestIdx=i; }
    lastTop = { label: classLabels[bestIdx], confidence: best };

    // show list w/ percentages (only place with %)
    renderListWithPerc(probs, bestIdx);

    tf.dispose([input,out]);
  }catch(e){
    document.getElementById('err').textContent = String(e?.message||e);
  }
  requestAnimationFrame(predictLoop);
}

/* ===== folder choosing & saving ===== */
async function chooseBaseFolder(){
  if(!window.showDirectoryPicker){
    document.getElementById('saveStatus').textContent='ブラウザがフォルダ保存に未対応です（Chrome/Edge 推奨）。';
    return;
  }
  baseDirHandle = await window.showDirectoryPicker();
  document.getElementById('saveStatus').textContent='保存先を設定しました。';
}
async function ensureSubdir(parent,name){ return parent.getDirectoryHandle(name,{create:true}); }
function yyyymmdd(){ const d=new Date(); return `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`; }

function captureFrameToBlob(){
  return new Promise(res=>{
    const c = document.getElementById('captureCanvas');
    const w=videoEl.videoWidth,h=videoEl.videoHeight;
    c.width=w; c.height=h;
    const g=c.getContext('2d'); g.drawImage(videoEl,0,0,w,h);
    c.toBlob(b=>res(b),'image/png',0.92);
  });
}
async function writeBlobTo(handle, filename, blob){
  const fh = await handle.getFileHandle(filename,{create:true});
  const w = await fh.createWritable(); await w.write(blob); await w.close();
}
function downloadFallback(filename, blob){
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}
async function saveCaptured(whereFolder, folderClass, baseLabel, conf, blob){
  const filename = `${yyyymmdd()}_${baseLabel}_${(conf*100).toFixed(0)}%.png`;
  try{
    if(baseDirHandle){
      const root = await ensureSubdir(baseDirHandle, whereFolder);
      const dir  = await ensureSubdir(root, folderClass);
      await writeBlobTo(dir, filename, blob);
      document.getElementById('saveStatus').textContent = `保存しました: ${whereFolder}/${folderClass}/${filename}`;
    }else{
      downloadFallback(`${whereFolder}_${folderClass}_${filename}`, blob);
      document.getElementById('saveStatus').textContent = `ダウンロードしました: ${whereFolder}/${folderClass}/${filename}`;
    }
  }catch(e){
    document.getElementById('saveStatus').textContent = `保存エラー: ${e.message||e}`;
  }finally{
    showAgree(false); showDisagree(false); showChoices(false);
  }
}

/* ===== capture flow with visible still ===== */
let capturedBlob=null, cameraJudgment=null, cameraConfidence=null;

function showAgree(v){ document.getElementById('agreeBtn').style.display = v?'':'none'; }
function showDisagree(v){ document.getElementById('disagreeBtn').style.display = v?'':'none'; }
function showChoices(v){
  document.getElementById('chooseBigLot').style.display = v?'':'none';
  document.getElementById('chooseCPress').style.display = v?'':'none';
  document.getElementById('chooseSnyders').style.display = v?'':'none';
}

document.getElementById('captureBtn').addEventListener('click', async ()=>{
  document.getElementById('saveStatus').textContent='';
  if(!videoEl || !lastTop){
    document.getElementById('saveStatus').textContent='まだ予測がありません。少し待ってください。';
    return;
  }

  // 1) capture blob of the current frame
  capturedBlob = await captureFrameToBlob();
  cameraJudgment = lastTop.label;
  cameraConfidence = lastTop.confidence;

  // 2) display the captured frame overlay so it doesn’t go black
  const url = URL.createObjectURL(capturedBlob);
  freezeImg.src = url;
  freezeImg.style.display = 'block';

  // 3) freeze: stop loop & stop camera
  isFrozen = true;
  running = false;
  if(videoEl.srcObject){
    for(const t of videoEl.srcObject.getVideoTracks()) t.stop();
    videoEl.srcObject = null;
  }
  videoEl.pause();
  document.getElementById('status').textContent='キャプチャ中（停止）';

  // 4) show Agree/Disagree
  showAgree(true); showDisagree(true); showChoices(false);
});

document.getElementById('agreeBtn').addEventListener('click', async ()=>{
  if(!capturedBlob) return;
  await saveCaptured('Agree', cameraJudgment, cameraJudgment, cameraConfidence, capturedBlob);
});
document.getElementById('disagreeBtn').addEventListener('click', ()=>{
  if(!capturedBlob) return;
  showChoices(true);
});
document.getElementById('chooseBigLot').addEventListener('click', async ()=>{
  if(!capturedBlob) return;
  await saveCaptured('Disagree','Big Lot',cameraJudgment,cameraConfidence,capturedBlob);
});
document.getElementById('chooseCPress').addEventListener('click', async ()=>{
  if(!capturedBlob) return;
  await saveCaptured('Disagree','C Press',cameraJudgment,cameraConfidence,capturedBlob);
});
document.getElementById('chooseSnyders').addEventListener('click', async ()=>{
  if(!capturedBlob) return;
  await saveCaptured('Disagree','Snyders',cameraJudgment,cameraConfidence,capturedBlob);
});

/* ===== top buttons ===== */
document.getElementById('chooseFolderBtn').addEventListener('click', chooseBaseFolder);

document.getElementById('startBtn').addEventListener('click', async ()=>{
  document.getElementById('err').textContent='';
  // unfreeze & hide overlay
  if(isFrozen){
    isFrozen=false;
    if(freezeImg.src){ URL.revokeObjectURL(freezeImg.src); }
    freezeImg.removeAttribute('src');
    freezeImg.style.display='none';
  }
  if(!videoEl || !videoEl.srcObject){
    await setupCamera();
  }
  if(!model) await loadModel();
  running = true;
  document.getElementById('status').textContent='推論中…';
  requestAnimationFrame(predictLoop);
});

/* ===== backend/init/cleanup ===== */
async function selectBackend(){
  try{ await tf.setBackend('webgl'); }catch(_){}
  if(tf.getBackend()!=='webgl'){ try{ await tf.setBackend('wasm'); }catch(_){} }
  await tf.ready();
  document.getElementById('backend').textContent = `backend: ${tf.getBackend()}`;
}

window.addEventListener('DOMContentLoaded', async ()=>{
  try{
    await selectBackend();
    await setupCamera();
    await loadModel();
    running = true;
    document.getElementById('status').textContent='推論中…';
    requestAnimationFrame(predictLoop);
  }catch(e){
    document.getElementById('err').textContent='初期化エラー: '+(e?.message||e);
  }
});

window.addEventListener('pagehide', ()=>{
  running=false;
  if(videoEl && videoEl.srcObject){ for(const t of videoEl.srcObject.getVideoTracks()) t.stop(); videoEl.srcObject=null; }
  tf.engine().disposeVariables(); tf.backend().dispose?.();
});
