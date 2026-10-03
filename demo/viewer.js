import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SCENES = ['bear', 'camel', 'cows', 'dog', 'horsejump-high', 'horsejump-low', 'libby', 'rhino'];
const ASSET_VERSION = '20261003-v2';
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const current = {
  name: SCENES.includes(params.get('scene')) ? params.get('scene') : 'horsejump-low',
  mode: params.get('mode') === 'foreground' ? 'foreground' : 'full_scene',
  meta: null, bytes: null, frame: 0, playing: false, lastTick: 0, loadSeq: 0,
  samples: null, pose: null, dirty: false, controller: null,
};
const cache = new Map();
const canvas = $('canvas');
const world = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.0001, 100);
const renderer = new THREE.WebGLRenderer({canvas, antialias: true, powerPreference: 'high-performance'});
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.setClearColor(0x111c25, 1);
let controls;
let lastSort = 0;
let decoded = null;

const geometry = new THREE.InstancedBufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0], 3));
geometry.setIndex([0,1,2, 0,2,3]);
const material = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, depthTest: false,
  uniforms: {uExtent: {value: 1}, uViewport: {value: new THREE.Vector2(1, 1)}, uPixelSigma: {value: 0.8}},
  vertexShader: `
    attribute vec3 iPosition; attribute vec3 iScale;
    attribute vec4 iRotation; attribute vec4 iColor;
    uniform float uExtent; uniform vec2 uViewport; uniform float uPixelSigma;
    varying vec2 vLocal; varying vec4 vColor;
    vec3 rotateVector(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
    vec2 projectAxis(vec3 mv, vec3 axis) {
      float d = max(0.00001, -mv.z);
      return vec2(projectionMatrix[0][0] * (axis.x / d + mv.x * axis.z / (d*d)),
                  projectionMatrix[1][1] * (axis.y / d + mv.y * axis.z / (d*d))) * uViewport * 0.5;
    }
    void main() {
      vec4 mv = modelViewMatrix * vec4(iPosition, 1.0);
      vec4 clip = projectionMatrix * mv;
      vec3 scale = exp(iScale) / uExtent;
      vec2 dx = projectAxis(mv.xyz, mat3(modelViewMatrix) * rotateVector(iRotation, vec3(scale.x,0.0,0.0)));
      vec2 dy = projectAxis(mv.xyz, mat3(modelViewMatrix) * rotateVector(iRotation, vec3(0.0,scale.y,0.0)));
      vec2 dz = projectAxis(mv.xyz, mat3(modelViewMatrix) * rotateVector(iRotation, vec3(0.0,0.0,scale.z)));
      float aa = dx.x*dx.x + dy.x*dy.x + dz.x*dz.x + uPixelSigma*uPixelSigma;
      float dd = dx.y*dx.y + dy.y*dy.y + dz.y*dz.y + uPixelSigma*uPixelSigma;
      float bb = dx.x*dx.y + dy.x*dy.y + dz.x*dz.y;
      float discriminant = sqrt(max(0.0, (aa-dd)*(aa-dd) + 4.0*bb*bb));
      float l1 = max(0.49, 0.5*(aa+dd+discriminant));
      float l2 = max(0.49, 0.5*(aa+dd-discriminant));
      vec2 e1 = abs(bb) > 0.00001 ? normalize(vec2(bb, l1-aa)) : (aa >= dd ? vec2(1.0,0.0) : vec2(0.0,1.0));
      vec2 e2 = vec2(-e1.y, e1.x);
      vec2 offset = 3.0 * (position.x*e1*min(16.0,sqrt(l1)) + position.y*e2*min(16.0,sqrt(l2)));
      gl_Position = vec4(clip.xy + offset*2.0/uViewport*clip.w, clip.z, clip.w);
      if (mv.z >= -0.00001) gl_Position = vec4(2.0,2.0,2.0,1.0);
      vLocal = position.xy; vColor = iColor;
    }`,
  fragmentShader: `
    varying vec2 vLocal; varying vec4 vColor;
    void main() { float r = dot(vLocal,vLocal); if(r > 1.0) discard;
      float alpha = exp(-4.5*r) * vColor.a; if(alpha < 0.003) discard;
      gl_FragColor = vec4(vColor.rgb,alpha);
    }`,
});
const cloud = new THREE.Mesh(geometry, material);
cloud.frustumCulled = false;
cloud.visible = false;
world.add(cloud);

function quantile(values, fraction) {
  const sorted = values.slice().sort((a,b) => a-b);
  return sorted[Math.min(sorted.length-1, Math.floor((sorted.length-1)*fraction))];
}

function sampleViews(meta, bytes) {
  return [...new Set([0, Math.floor(meta.frames.length/2), meta.frames.length-1])].map(index => {
    const record = meta.frames[index];
    const data = new DataView(bytes, record.offset, record.count * meta.stride);
    const points = [];
    for (let i=0; i<record.count; i+=4) {
      const b = i*meta.stride;
      const p = [data.getFloat32(b,true), data.getFloat32(b+4,true), data.getFloat32(b+8,true)];
      if(data.getUint8(b+15) >= 24 && p.every(v => Number.isFinite(v) && Math.abs(v)<3.95)) points.push(p);
    }
    if(!points.length) throw new Error('No visible scene samples');
    return points;
  });
}

function computePose() {
  const samples = current.samples;
  const points = samples.flat();
  if(current.mode === 'full_scene') {
    // The asset normalization moves the source camera away from world origin.
    const origin = current.meta.center.map(v => -v/current.meta.extent);
    const visible = points.filter(p => origin[2]-p[2] > 0.00001);
    const depths = visible.map(p => origin[2]-p[2]);
    const u = visible.map(p => (p[0]-origin[0])/(origin[2]-p[2]));
    const v = visible.map(p => (p[1]-origin[1])/(origin[2]-p[2]));
    const halfWidth = Math.max(Math.abs(quantile(u,.005)), Math.abs(quantile(u,.995)));
    const halfHeight = Math.max(Math.abs(quantile(v,.005)), Math.abs(quantile(v,.995)));
    const depth = quantile(depths,.5);
    const fov = THREE.MathUtils.radToDeg(2*Math.atan(Math.max(halfHeight,halfWidth/camera.aspect)*1.04));
    return {position: origin, target: [origin[0],origin[1],origin[2]-depth], up: [0,1,0],
      fov, near: Math.max(0.00001,quantile(depths,.01)*.03), distance: depth};
  }
  // Foreground outputs use Z-up object space, with their front facing +Y.
  const low = [0,1,2].map(axis => Math.min(...samples.map(ps => quantile(ps.map(p=>p[axis]),.01))));
  const high = [0,1,2].map(axis => Math.max(...samples.map(ps => quantile(ps.map(p=>p[axis]),.99))));
  const target = low.map((v,i) => (v+high[i])/2);
  target[2] -= (high[2]-low[2])*.05;
  const tanHalf = Math.tan(THREE.MathUtils.degToRad(21));
  const required = samples.map(ps => quantile(ps.map(p => (p[1]-target[1]) +
    Math.max(Math.abs(p[0]-target[0])/(tanHalf*camera.aspect),Math.abs(p[2]-target[2])/tanHalf)),.99));
  const distance = Math.max(.1, Math.max(...required)*1.35);
  return {position: [target[0],target[1]+distance,target[2]], target, up: [0,0,1], fov:42, near:distance*.001, distance};
}

function resetView() {
  if(!current.samples) return;
  // Recreate controls when changing up-axis; clear the previous orbit's damping.
  controls?.dispose();
  current.pose = computePose();
  camera.up.fromArray(current.pose.up);
  camera.position.fromArray(current.pose.position);
  camera.fov = current.pose.fov;
  camera.near = current.pose.near;
  camera.far = Math.max(100,current.pose.distance*20);
  camera.zoom = 1;
  camera.lookAt(new THREE.Vector3().fromArray(current.pose.target));
  camera.updateProjectionMatrix();
  controls = new OrbitControls(camera,canvas);
  controls.target.fromArray(current.pose.target);
  controls.enableDamping = true;
  controls.dampingFactor = .12;
  controls.enableZoom = false;
  controls.minDistance = current.pose.distance*.15;
  controls.maxDistance = current.pose.distance*8;
  controls.addEventListener('change', () => {current.dirty=true; updateZoomLabel();});
  controls.update();
  $('zoom').value = '100';
  updateZoomLabel();
  current.dirty = true;
  canvas.dataset.camera = JSON.stringify(current.pose);
}

function updateZoomLabel() {
  const effective = camera.zoom;
  $('zoom-value').value = Math.round(effective*100)+'%';
  $('zoom').value = String(Math.round(effective*100));
}

function halfToFloat(h) {
  const sign = h&0x8000 ? -1 : 1, exponent = h>>10&31, fraction = h&1023;
  if(exponent===0) return sign*2**-14*fraction/1024;
  if(exponent===31) return fraction ? NaN : sign*Infinity;
  return sign*2**(exponent-15)*(1+fraction/1024);
}

function updateFrame(syncVideo=true) {
  if(!current.meta) return;
  const record = current.meta.frames[current.frame];
  const data = new DataView(current.bytes,record.offset,record.count*current.meta.stride);
  decoded = {count:record.count, position:new Float32Array(record.count*3), scale:new Float32Array(record.count*3),
    rotation:new Float32Array(record.count*4), color:new Float32Array(record.count*4)};
  for(let i=0; i<record.count; i++) {
    const b = i*current.meta.stride;
    for(let j=0; j<3; j++) {
      decoded.position[i*3+j]=data.getFloat32(b+j*4,true);
      decoded.scale[i*3+j]=halfToFloat(data.getUint16(b+16+j*2,true));
    }
    for(let j=0; j<4; j++) {
      decoded.color[i*4+j]=data.getUint8(b+12+j)/255;
      decoded.rotation[i*4+j]=data.getInt16(b+22+j*2,true)/32767;
    }
  }
  current.dirty = true;
  sortSplats();
  $('frame').value = String(current.frame);
  $('frame-value').value = `${current.frame+1} / ${current.meta.frames.length}`;
  canvas.dataset.frame = String(current.frame);
  if(syncVideo) seekVideo();
}

function sortSplats() {
  if(!decoded) return;
  camera.updateMatrixWorld();
  const m = camera.matrixWorldInverse.elements;
  const depth = new Float32Array(decoded.count);
  for(let i=0;i<decoded.count;i++) depth[i]=m[2]*decoded.position[i*3]+m[6]*decoded.position[i*3+1]+m[10]*decoded.position[i*3+2];
  const order = Uint32Array.from({length:decoded.count},(_,i)=>i);
  order.sort((a,b)=>depth[a]-depth[b]);
  for(const [attribute,key,size] of [['iPosition','position',3],['iScale','scale',3],['iRotation','rotation',4],['iColor','color',4]]) {
    let buffer = geometry.getAttribute(attribute);
    if(!buffer || buffer.count < decoded.count) {
      // Keep one capacity for both modes; the renderer caches the instance limit.
      buffer = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(30000,decoded.count)*size),size);
      buffer.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(attribute,buffer);
    }
    for(let i=0;i<decoded.count;i++) for(let j=0;j<size;j++) buffer.array[i*size+j]=decoded[key][order[i]*size+j];
    buffer.needsUpdate = true;
  }
  geometry.instanceCount = decoded.count;
  cloud.visible = $('show-output').checked;
  const coverage=current.mode==='full_scene'?.68:.35;
  material.uniforms.uPixelSigma.value = Math.max(.7, Math.sqrt(canvas.clientWidth*canvas.clientHeight/decoded.count)*coverage);
  current.dirty=false;
}

function seekVideo() {
  const video = $('input-video');
  if(!current.meta || !Number.isFinite(video.duration)) return;
  const time = (current.meta.frames[current.frame].frame ?? current.frame)/(current.meta.fps || 12);
  video.currentTime = Math.min(time,Math.max(0,video.duration-.05));
}

function setPlaying(playing) {
  current.playing = playing && !!current.meta;
  current.lastTick = performance.now();
  $('play').textContent = current.playing ? 'Ⅱ' : '▶';
  $('play').setAttribute('aria-label',current.playing?'Pause sequence':'Play sequence');
  const video = $('input-video');
  if(current.playing) {video.playbackRate=Number($('fps').value)/(current.meta.fps||12); video.play().catch(()=>{});}
  else {video.pause(); seekVideo();}
}

function setLoading(message,error=false) {
  $('loading-indicator').hidden=false;
  $('loading-indicator').classList.toggle('error',error);
  $('load-message').textContent=message;
  $('retry').hidden=!error;
}

async function fetchAsset(url,signal) {
  const response = await fetch(url+'?v='+ASSET_VERSION,{signal});
  if(!response.ok) throw new Error(`Asset request returned ${response.status}`);
  return response;
}

async function downloadFrames(base,meta,signal,seq) {
  const response = await fetchAsset(base+'/frames.bin',signal);
  const last=meta.frames.at(-1), expected=last.offset+last.count*meta.stride;
  if(!response.body) return response.arrayBuffer();
  const bytes = new Uint8Array(expected);
  const reader = response.body.getReader();
  let offset=0;
  while(true) {
    const {done,value}=await reader.read();
    if(done) break;
    if(offset+value.length>expected) throw new Error('Scene payload has unexpected length');
    bytes.set(value,offset); offset+=value.length;
    if(seq===current.loadSeq) $('load-message').textContent=`Loading ${current.name} · ${Math.round(offset/expected*100)}%`;
  }
  if(offset!==expected) throw new Error('Scene payload is incomplete');
  return bytes.buffer;
}

async function loadScene(name,mode=current.mode,preserveFrame=false) {
  if(!SCENES.includes(name)) return;
  const oldFrame = preserveFrame ? current.frame : 0;
  setPlaying(false);
  current.controller?.abort();
  const controller = new AbortController();
  current.controller=controller;
  const seq=++current.loadSeq;
  current.name=name; current.mode=mode; current.meta=null; current.bytes=null; current.samples=null;
  decoded=null; cloud.visible=false;
  $('play').disabled=true; $('frame').disabled=true; $('reset-view').disabled=true; $('zoom').disabled=true;
  $('stat').textContent='';
  $('scene-label').textContent=name;
  $('view-label').textContent=mode==='full_scene'?'Full scene':'Foreground';
  $('stage').dataset.mode=mode; $('stage').dataset.state='loading';
  document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
  document.querySelectorAll('.scene-button').forEach(b=>{b.classList.toggle('active',b.dataset.scene===name);b.setAttribute('aria-pressed',String(b.dataset.scene===name));});
  const search=new URLSearchParams(location.search); search.set('scene',name); search.set('mode',mode);
  history.replaceState(null,'',location.pathname+'?'+search);
  setLoading('Loading '+name+'…');
  try {
    const key=mode+'/'+name;
    let asset=cache.get(key);
    if(!asset) {
      const base=(mode==='full_scene'?'./assets/':'./assets_foreground/')+name;
      const meta=await (await fetchAsset(base+'/meta.json',controller.signal)).json();
      if(meta.format!=='web-gaussian-v1' || meta.stride!==30 || !meta.frames.length) throw new Error('Unsupported scene format');
      const bytes=await downloadFrames(base,meta,controller.signal,seq);
      asset={meta,bytes,samples:sampleViews(meta,bytes)};
      if(seq!==current.loadSeq) return;
      cache.set(key,asset);
      while(cache.size>3) cache.delete(cache.keys().next().value);
    } else { cache.delete(key); cache.set(key,asset); }
    if(seq!==current.loadSeq) return;
    current.meta=asset.meta; current.bytes=asset.bytes; current.samples=asset.samples;
    current.frame=Math.min(oldFrame,asset.meta.frames.length-1);
    material.uniforms.uExtent.value=asset.meta.extent||1;
    resetView(); updateFrame();
    $('frame').max=String(asset.meta.frames.length-1);
    $('play').disabled=false; $('frame').disabled=false; $('reset-view').disabled=false; $('zoom').disabled=false;
    const video=$('input-video');
    video.poster='./thumbs/'+name+'.png';
    const src=new URL('./assets/'+name+'_input.mp4',location.href).href;
    if(video.src!==src) video.src=src;
    seekVideo();
    $('loading-indicator').hidden=true;
    $('stat').textContent=`${asset.meta.frames.length} frames · ${mode==='full_scene'?'Full scene':'Foreground'} output`;
    $('stage').dataset.state='ready';
  } catch(error) {
    if(seq!==current.loadSeq || error.name==='AbortError') return;
    console.error(error);
    $('stage').dataset.state='error';
    setLoading('Could not load this output. Check your connection and retry.',true);
  }
}

function resize() {
  const r=$('stage').getBoundingClientRect();
  const oldAspect=camera.aspect;
  renderer.setSize(r.width,r.height,false);
  material.uniforms.uViewport.value.set(r.width,r.height);
  camera.aspect=r.width/r.height;
  camera.updateProjectionMatrix();
  if(current.samples && Math.abs(oldAspect-camera.aspect)>.01) resetView();
  current.dirty=true;
  clampPanel();
}

const panel=$('control-panel');
function clampPanel() {
  const stage=$('stage');
  const left=Math.min(Math.max(8,parseFloat(panel.style.left)||16), Math.max(8,stage.clientWidth-panel.offsetWidth-8));
  const top=Math.min(Math.max(8,parseFloat(panel.style.top)||16), Math.max(8,stage.clientHeight-panel.offsetHeight-90));
  panel.style.left=left+'px'; panel.style.top=top+'px';
}
function sizePanel(width,height) {
  const stage=$('stage'), left=parseFloat(panel.style.left)||16, top=parseFloat(panel.style.top)||16;
  panel.style.width=Math.max(188,Math.min(340,stage.clientWidth-left-8,width))+'px';
  panel.style.height=Math.max(224,Math.min(stage.clientHeight-top-92,height))+'px';
}
function pointerGesture(element,onStart,onMove) {
  element.addEventListener('pointerdown',event=>{
    if(event.button!==0 || event.target.closest('.icon-button')) return;
    event.preventDefault();
    const state=onStart(event);
    if(!state) return;
    element.setPointerCapture(event.pointerId);
    const move=e=>onMove(e,state);
    const end=()=>{element.removeEventListener('pointermove',move);element.removeEventListener('pointerup',end);element.removeEventListener('pointercancel',end);};
    element.addEventListener('pointermove',move);element.addEventListener('pointerup',end);element.addEventListener('pointercancel',end);
  });
}
pointerGesture($('panel-drag'),event=>panel.classList.contains('minimized')?null:
  ({x:event.clientX,y:event.clientY,left:panel.offsetLeft,top:panel.offsetTop}), (event,state)=>{
  panel.style.left=state.left+event.clientX-state.x+'px'; panel.style.top=state.top+event.clientY-state.y+'px';clampPanel();
});
pointerGesture($('panel-resize'),event=>({x:event.clientX,y:event.clientY,width:panel.offsetWidth,height:panel.offsetHeight}),
  (event,state)=>sizePanel(state.width+event.clientX-state.x,state.height+event.clientY-state.y));
$('panel-resize').addEventListener('keydown',event=>{
  if(!event.key.startsWith('Arrow')) return;
  event.preventDefault();
  sizePanel(panel.offsetWidth+(event.key==='ArrowRight'?10:event.key==='ArrowLeft'?-10:0),panel.offsetHeight+(event.key==='ArrowDown'?10:event.key==='ArrowUp'?-10:0));
});
$('panel-toggle').onclick=()=>{
  const minimized=panel.classList.toggle('minimized');
  $('panel-toggle').textContent=minimized?'☰':'−';
  $('panel-toggle').setAttribute('aria-label',minimized?'Expand controls':'Collapse controls');
  $('panel-toggle').setAttribute('aria-expanded',String(!minimized));
  $('panel-toggle').title=minimized?'Expand controls':'Collapse controls';
  clampPanel();
};
if(matchMedia('(max-width:600px)').matches) $('panel-toggle').click();

SCENES.forEach(name=>{
  const button=document.createElement('button');
  button.type='button';button.className='scene-button';button.dataset.scene=name;button.title='Load '+name;
  button.innerHTML=`<img loading="lazy" src="./thumbs/${name}.png" alt=""><span>${name}</span>`;
  button.onclick=()=>loadScene(name);$('scene-grid').appendChild(button);
});
document.querySelectorAll('[data-mode]').forEach(button=>button.onclick=()=>{
  if(button.dataset.mode!==current.mode) loadScene(current.name,button.dataset.mode,true);
});
$('play').onclick=()=>setPlaying(!current.playing);
$('frame').oninput=()=>{setPlaying(false);current.frame=Number($('frame').value);updateFrame();};
$('fps').onchange=()=>{if(current.playing) $('input-video').playbackRate=Number($('fps').value)/(current.meta.fps||12);};
$('reset-view').onclick=resetView;
$('zoom').oninput=()=>{
  camera.zoom=Number($('zoom').value)/100;
  camera.updateProjectionMatrix();updateZoomLabel();current.dirty=true;
};
canvas.addEventListener('wheel',event=>{
  event.preventDefault();
  if(!current.meta) return;
  camera.zoom=Math.max(.6,Math.min(1.8,camera.zoom*Math.exp(-event.deltaY*.001)));
  camera.updateProjectionMatrix();updateZoomLabel();current.dirty=true;
},{passive:false});
$('show-source').onchange=()=>{$('source-preview').hidden=!$('show-source').checked;};
$('hide-source').onclick=()=>{$('show-source').checked=false;$('source-preview').hidden=true;};
$('light-bg').onchange=()=>renderer.setClearColor($('light-bg').checked?0xe8eef2:0x111c25,1);
$('show-output').onchange=()=>cloud.visible=$('show-output').checked && !!current.meta;
$('input-video').onloadedmetadata=seekVideo;
$('retry').onclick=()=>loadScene(current.name);
window.addEventListener('keydown',event=>{
  if(event.target.matches('input,select,button,summary')) return;
  if(event.key===' '){event.preventDefault();setPlaying(!current.playing);}
  else if(event.key.toLowerCase()==='r' || event.key==='0') resetView();
  else if(current.meta && ['ArrowRight','ArrowLeft'].includes(event.key)){
    event.preventDefault();setPlaying(false);
    current.frame=(current.frame+(event.key==='ArrowRight'?1:-1)+current.meta.frames.length)%current.meta.frames.length;updateFrame();
  }
});

function animate(now) {
  requestAnimationFrame(animate);
  controls?.update();
  const interval=1000/Number($('fps').value);
  if(current.playing && current.meta && now-current.lastTick>=interval) {
    const step=Math.max(1,Math.floor((now-current.lastTick)/interval));
    current.lastTick=now;
    current.frame=(current.frame+step)%current.meta.frames.length;
    updateFrame(false);
    const video=$('input-video');
    if(Number.isFinite(video.duration) && Math.abs(video.currentTime-current.frame/(current.meta.fps||12))>.3) seekVideo();
  } else if(current.dirty && decoded && now-lastSort>100) {sortSplats();lastSort=now;}
  renderer.render(world,camera);
}
resize();
new ResizeObserver(resize).observe($('stage'));
loadScene(current.name,current.mode);
requestAnimationFrame(animate);
