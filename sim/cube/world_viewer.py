"""Standalone body-map viewer (increment 5) — drop-a-cube, see the systemic thesis run.

Writes a self-contained HTML file: two live cubes (oral + artery) rendered from REAL engine point
clouds, joined by the bloodstream (particles flow ∝ the live OMV efflux), with per-cube ψ panels,
time-playback of the coupled dose-response, a cell legend, and an ask-anything panel wired to the
grounded Q&A (answers are the engine's real numbers; a live-Claude hook can replace the static text).
Three.js is inlined (offline-capable). Medical-grade palette — no punk glow.
"""

from __future__ import annotations

import json

from sim.cube.viewer import _three_bundle
from sim.cube.world_export import build_world


def write_world_viewer(path: str, *, dim: int = 26, steps: int = 160, frames: int = 6,
                       n_points: int = 1100, k: float = 2.5) -> str:
    world = build_world(dim=dim, steps=steps, frames=frames, n_points=n_points, k=k)
    html = (_TEMPLATE
            .replace("/*__THREE__*/", _three_bundle())
            .replace("/*__DATA__*/", json.dumps(world)))
    with open(path, "w") as fh:
        fh.write(html)
    return path


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OVN Body-Map — oral → systemic</title>
<style>
  :root{--bg0:#0d1217;--bg1:#141c24;--ink:#dde7ef;--dim:#8aa0b2;--line:#27384a;--accent:#7fd3e0;--blood:#c0392b;}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 800px at 50% -10%,#16212b,var(--bg0));
       color:var(--ink);font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
  #c{position:fixed;inset:0;display:block}
  .panel{position:fixed;background:rgba(16,24,32,.82);border:1px solid var(--line);border-radius:10px;
         padding:11px 13px;backdrop-filter:blur(7px)}
  #title{top:14px;left:50%;transform:translateX(-50%);text-align:center;max-width:680px}
  #title b{letter-spacing:.4px}#title .s{color:var(--dim);font-size:11.5px;margin-top:3px}
  #psiO{top:90px;left:14px;width:212px}#psiA{top:90px;right:14px;width:232px}
  .psi h4{margin:0 0 6px;font-size:12px;color:var(--accent);font-weight:600}
  .row{display:flex;justify-content:space-between;gap:8px;border-top:1px solid var(--line);padding:2px 0}
  .row:first-of-type{border-top:none}.row b{font-variant-numeric:tabular-nums}
  #legend{bottom:14px;left:14px;max-width:230px}#legend .lg{display:flex;align-items:center;gap:7px;margin:2px 0}
  .dot{width:10px;height:10px;border-radius:50%;flex:none}
  #ctl{bottom:14px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:12px;width:440px}
  #ctl input[type=range]{flex:1}
  button{background:#1b2a38;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:5px 11px;cursor:pointer}
  button:hover{border-color:var(--accent)}button.on{background:#21465a;border-color:var(--accent)}
  #qa{bottom:14px;right:14px;width:360px}
  #qa .qs{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:7px}
  #qa .qs button{font-size:11px;padding:4px 7px}
  #qa .ans{color:var(--ink);font-size:12px;min-height:54px;border-top:1px solid var(--line);padding-top:7px}
  #qa .who{display:flex;gap:6px;margin-bottom:7px}
  .cap{color:var(--dim)}
</style></head><body>
<canvas id="c"></canvas>
<div id="title" class="panel"><b>OVN BODY-MAP — oral biofilm → bloodstream → artery</b>
  <div class="s">Two live information-primary cubes (Epoch-4). The oral cube's heme-induced <b>OMV efflux</b>
  travels the bloodstream and inflames the artery's endothelium → foam cells. Everything shown is real
  engine output. <span class="cap">GCN roles R²≈0.27 · binding curated/ESMFold-pending</span></div></div>
<div id="psiO" class="panel psi"><h4>ORAL ψ</h4><div id="psiObody"></div></div>
<div id="psiA" class="panel psi"><h4>ARTERY ψ</h4><div id="psiAbody"></div></div>
<div id="legend" class="panel"><div id="legbody"></div></div>
<div id="ctl" class="panel"><button id="play">▶ play</button>
  <input id="frame" type="range" min="0" value="0" step="1"><span id="tlab" class="cap">t=0h</span></div>
<div id="qa" class="panel"><div class="who"><button id="whoO" class="on">oral</button><button id="whoA">artery</button>
  <span class="cap" style="align-self:center">— ask anything</span></div>
  <div class="qs" id="qbtns"></div><div class="ans" id="ans">Pick a question.</div></div>
/*__THREE__*/
<script>
const W=/*__DATA__*/;
const cv=document.getElementById('c');
if(!window.THREE){document.getElementById('ans').textContent='3-D engine failed to load (needs internet the first time it is generated).';}
const renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x0d1217,1);
const scene=new THREE.Scene();scene.fog=new THREE.FogExp2(0x0d1217,0.06);
const cam=new THREE.PerspectiveCamera(46,1,0.1,100);cam.position.set(0,1.4,7.6);
function resize(){renderer.setSize(innerWidth,innerHeight);cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();}
addEventListener('resize',resize);resize();
scene.add(new THREE.AmbientLight(0x88aacc,0.7));
const dl=new THREE.DirectionalLight(0xffffff,0.5);dl.position.set(3,6,5);scene.add(dl);

// soft round sprite for points
function disc(){const s=64,cv=document.createElement('canvas');cv.width=cv.height=s;const g=cv.getContext('2d');
  const r=g.createRadialGradient(s/2,s/2,1,s/2,s/2,s/2);r.addColorStop(0,'rgba(255,255,255,1)');
  r.addColorStop(.5,'rgba(255,255,255,.7)');r.addColorStop(1,'rgba(255,255,255,0)');
  g.fillStyle=r;g.fillRect(0,0,s,s);const t=new THREE.Texture(cv);t.needsUpdate=true;return t;}
const DISC=disc();
function label(txt){const cv=document.createElement('canvas');cv.width=256;cv.height=64;const g=cv.getContext('2d');
  g.fillStyle='rgba(0,0,0,0)';g.fillRect(0,0,256,64);g.font='600 30px sans-serif';g.fillStyle='#cfe2ee';
  g.textAlign='center';g.fillText(txt,128,42);const t=new THREE.Texture(cv);t.needsUpdate=true;
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:t,transparent:true}));sp.scale.set(2.0,0.5,1);return sp;}

const S=2.2, OFF=2.05;
function makeCube(cd,originX){
  const grp=new THREE.Group();grp.position.x=originX;scene.add(grp);
  const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(S,S,S)),
      new THREE.LineBasicMaterial({color:0x2b3e52}));grp.add(box);
  const cols=cd.legend.map(l=>new THREE.Color(l.color));
  const geo=new THREE.BufferGeometry();
  const mat=new THREE.PointsMaterial({size:0.14,map:DISC,vertexColors:true,transparent:true,
      alphaTest:0.02,depthWrite:false,opacity:0.97,sizeAttenuation:true});
  const pts=new THREE.Points(geo,mat);grp.add(pts);
  const lab=label(cd.name+' — '+cd.tissue);lab.position.set(0,S/2+0.45,0);grp.add(lab);
  return {grp,geo,pts,cols,cd};
}
function setCubeFrame(cube,f){
  const fr=cube.cd.frames[f]||[];const pos=new Float32Array(fr.length*3),col=new Float32Array(fr.length*3);
  for(let i=0;i<fr.length;i++){const p=fr[i];
    pos[i*3]=(p[0]-0.5)*S; pos[i*3+1]=(p[2]-0.5)*S; pos[i*3+2]=(p[1]-0.5)*S;  // k(depth)→vertical
    const c=cube.cols[p[3]]||new THREE.Color('#888');col[i*3]=c.r;col[i*3+1]=c.g;col[i*3+2]=c.b;}
  cube.geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  cube.geo.setAttribute('color',new THREE.BufferAttribute(col,3));cube.geo.attributes.position.needsUpdate=true;
}
const oral=makeCube(W.cubes[0],-OFF), artery=makeCube(W.cubes[1],OFF);

// bloodstream
const curve=new THREE.QuadraticBezierCurve3(new THREE.Vector3(-OFF+S/2-0.1,0.5,0),
   new THREE.Vector3(0,1.7,0),new THREE.Vector3(OFF-S/2+0.1,0.5,0));
const tube=new THREE.Mesh(new THREE.TubeGeometry(curve,40,0.03,8,false),
   new THREE.MeshBasicMaterial({color:0x7a2230,transparent:true,opacity:0.5}));scene.add(tube);
const NP=90,bgeo=new THREE.BufferGeometry();bgeo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(NP*3),3));
const bmat=new THREE.PointsMaterial({size:0.09,map:DISC,color:0xe0584a,transparent:true,depthWrite:false,opacity:0.9});
const blood=new THREE.Points(bgeo,bmat);scene.add(blood);const bph=new Float32Array(NP).map((_,i)=>i/NP);
const lbl=label('bloodstream (OMV)');lbl.position.set(0,2.05,0);lbl.scale.set(2.4,0.6,1);scene.add(lbl);

// ψ panels
const PSI={oral:{el:'psiObody',keys:[['dysbiosis','dysbiosis index'],['omv_flux','OMV flux'],['NO','NO output'],['biomass','biomass']]},
  artery:{el:'psiAbody',keys:[['endothelial_activation','endothelial activation'],['foam_fraction','foam fraction'],
     ['plaque_burden','plaque burden'],['inflammation_load','inflammatory load']]}};
function psiRows(cd,spec,f){const p=cd.psi[f];return spec.keys.map(([k,lab])=>
  `<div class="row"><span class="cap">${lab}</span><b>${p[k]}</b></div>`).join('');}
function setFrame(f){
  setCubeFrame(oral,f);setCubeFrame(artery,f);
  document.getElementById('psiObody').innerHTML=psiRows(W.cubes[0],PSI.oral,f);
  document.getElementById('psiAbody').innerHTML=psiRows(W.cubes[1],PSI.artery,f);
  document.getElementById('tlab').textContent='t='+W.times[f]+'h  ·  OMV efflux '+W.efflux[f];
  CUR=f;
}
// legend (both cubes)
(function(){let h='';[['ORAL',W.cubes[0]],['ARTERY',W.cubes[1]]].forEach(([t,cd])=>{
  h+=`<div style="color:var(--accent);margin:3px 0 2px">${t}</div>`;
  cd.legend.forEach(l=>{h+=`<div class="lg"><span class="dot" style="background:${l.color}"></span>${l.id} <span class="cap">${l.label}</span></div>`;});});
  document.getElementById('legbody').innerHTML=h;})();

// Q&A
let WHO='oral';
function renderQA(){const qa=W.qa[WHO];let h='';qa.forEach((x,i)=>h+=`<button data-i="${i}">${x.q}</button>`);
  document.getElementById('qbtns').innerHTML=h;
  document.querySelectorAll('#qbtns button').forEach(b=>b.onclick=()=>{
    document.getElementById('ans').textContent=W.qa[WHO][+b.dataset.i].a;});
  document.getElementById('ans').textContent='Pick a question.';}
document.getElementById('whoO').onclick=()=>{WHO='oral';whoO.classList.add('on');whoA.classList.remove('on');renderQA();};
document.getElementById('whoA').onclick=()=>{WHO='artery';whoA.classList.add('on');whoO.classList.remove('on');renderQA();};
renderQA();

// playback
const slider=document.getElementById('frame');slider.max=W.n_frames-1;let CUR=0,PLAY=false;
slider.oninput=()=>setFrame(+slider.value);
document.getElementById('play').onclick=function(){PLAY=!PLAY;this.classList.toggle('on');this.textContent=PLAY?'❚❚ pause':'▶ play';};
setFrame(0);

// interaction: drag-orbit
let yaw=0.5,pit=0.15,drag=false,px=0,py=0;
cv.addEventListener('pointerdown',e=>{drag=true;px=e.clientX;py=e.clientY});
addEventListener('pointerup',()=>drag=false);
addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-px)*0.005;pit=Math.max(-0.6,Math.min(0.9,pit+(e.clientY-py)*0.004));px=e.clientX;py=e.clientY;});
let tprev=0,acc=0;
function frameTick(dt){acc+=dt;if(PLAY&&acc>0.9){acc=0;CUR=(CUR+1)%W.n_frames;slider.value=CUR;setFrame(CUR);}}
function loop(t){const dt=(t-tprev)/1000;tprev=t;frameTick(dt);
  const R=8.2;cam.position.set(Math.sin(yaw)*R*Math.cos(pit),1.4+Math.sin(pit)*R,Math.cos(yaw)*R*Math.cos(pit));
  cam.lookAt(0,0.5,0);
  const e=W.efflux[CUR],en=Math.min(1,e/0.04);bmat.size=0.06+0.07*en;bmat.opacity=0.45+0.5*en;
  const pos=blood.geometry.attributes.position.array;
  for(let i=0;i<NP;i++){bph[i]=(bph[i]+dt*(0.05+0.16*en))%1;const v=curve.getPoint(bph[i]);
    pos[i*3]=v.x;pos[i*3+1]=v.y;pos[i*3+2]=v.z;}
  blood.geometry.attributes.position.needsUpdate=true;
  oral.grp.rotation.y=artery.grp.rotation.y=0; // cubes static; camera orbits
  renderer.render(scene,cam);requestAnimationFrame(loop);}
requestAnimationFrame(loop);
</script></body></html>"""

