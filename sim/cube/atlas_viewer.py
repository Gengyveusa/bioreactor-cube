"""Body-map atlas viewer — navigate a body, drop into one polished cube at a time.

Preserves the single-cube presentation (surfaces + info box, glow points by group, gradient slices,
8-layer cake, time-playback) and adds: a body-map navigator (click an organ → load that tissue),
a ψ readout, the GNN network panel, and the grounded ask-anything Q&A. One cube in focus at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

from sim.cube.atlas import build_atlas
from sim.cube.viewer import _three_bundle


def write_atlas_viewer(path: str, *, oral_dim: int = 40, art_dim: int = 24) -> str:
    data = build_atlas(oral_dim=oral_dim, art_dim=art_dim)
    html = (_TEMPLATE
            .replace("/*__THREE__*/", _three_bundle())
            .replace("/*__DATA__*/",
                     "const ATLAS=" + json.dumps(data, separators=(",", ":")) + ";"))
    Path(path).write_text(html, encoding="utf-8")
    return path


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OVN Atlas — the body-map cube</title>
<meta name="robots" content="noindex,nofollow">
<meta name="description" content="An interactive 3-D atlas of the oral–systemic axis: how oral inflammation and pathogens leave the mouth and reach the artery, brain, and liver — modeled tissue by tissue.">
<meta property="og:type" content="website"><meta property="og:site_name" content="OVN Nexus"><meta property="og:title" content="OVN Atlas — the body-map cube"><meta property="og:description" content="An interactive 3-D atlas of the oral–systemic axis: oral inflammation and pathogens, modeled tissue by tissue."><meta property="og:url" content="https://ovnnexus.org/ovn_atlas.html">
<meta name="twitter:card" content="summary"><meta name="twitter:title" content="OVN Atlas — the body-map cube"><meta name="twitter:description" content="An interactive 3-D atlas of the oral–systemic axis, modeled tissue by tissue.">
<style>
  :root{--bg:#070a0f;--panel:rgba(18,24,34,.82);--line:rgba(255,255,255,.09);--tx:#e9eef5;--tx2:#9fb0c3;--tx3:#5f6f80;--accent:#2bd4e8;}
  *{box-sizing:border-box;}html,body{margin:0;height:100%;background:var(--bg);color:var(--tx);font-family:Inter,system-ui,Segoe UI,sans-serif;overflow:hidden;}
  #c{position:fixed;inset:0;width:100%;height:100%;z-index:0;touch-action:pan-y;}
  .pan{position:fixed;background:var(--panel);backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px);border:.5px solid var(--line);border-radius:14px;}
  #top{top:14px;left:50%;transform:translateX(-50%);padding:9px 18px;text-align:center;z-index:22;max-width:calc(100vw - 672px);}
  #top h1{margin:0;font-size:15.5px;font-weight:500;}#top .sub{font-size:11.5px;color:var(--tx2);}
  /* Each side of the screen is ONE flex column pinned top:14 -> bottom:14. The column has a
     definite height, so its children divide that height instead of being independently anchored
     to opposite edges -- which is what made #qa ride up over #side and #ctl over #nav. */
  .col{position:fixed;top:14px;bottom:14px;display:flex;flex-direction:column;gap:12px;pointer-events:none;z-index:20;}
  .col>.pan{position:static;pointer-events:auto;min-height:0;flex:1 1 auto;overflow:auto;}
  #colL{left:14px;width:320px;align-items:flex-start;}
  #colR{right:14px;width:296px;}
  #nav{width:190px;padding:12px;flex:0 0 auto;max-height:56%;}
  #nav .h{font-size:11px;color:var(--tx2);letter-spacing:.3px;margin:0 0 8px;}
  #bsvg{width:100%;height:clamp(118px,26vh,230px);display:block;}
  #ctl{width:320px;padding:13px;}
  #side{width:296px;padding:13px;}
  #qa{width:296px;padding:12px;flex:0 0 auto;max-height:46%;}
  .glbl{font-size:10.5px;font-weight:500;color:var(--tx2);margin:2px 0 6px;display:block;letter-spacing:.3px;}
  .grp{margin:0 0 12px;}
  .btn{display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:5px 9px;margin:0 5px 5px 0;border-radius:8px;border:.5px solid var(--line);background:rgba(255,255,255,.03);color:var(--tx2);cursor:pointer;transition:.14s;}
  .btn .d{width:9px;height:9px;border-radius:50%;}.btn.on{background:rgba(255,255,255,.10);color:var(--tx);border-color:rgba(255,255,255,.28);}
  .btn:not(.on) .d{opacity:.32;}
  select,input[type=range]{font:inherit;color:var(--tx);}select{background:rgba(255,255,255,.05);border:.5px solid var(--line);border-radius:8px;padding:4px 7px;font-size:12px;}
  input[type=range]{accent-color:var(--accent);}
  #exp{font-size:12px;line-height:1.55;color:var(--tx2);background:rgba(255,255,255,.04);border:.5px solid var(--line);border-radius:10px;padding:10px 12px;margin:0 0 11px;}
  #exp b{color:var(--tx);font-weight:500;}
  .h4{font-size:11.5px;font-weight:600;margin:11px 0 7px;color:var(--accent);}
  .key{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--tx2);margin:0 0 4px;}.key .d{width:10px;height:10px;border-radius:50%;flex:none;}
  .row{display:flex;justify-content:space-between;gap:8px;font-size:12px;border-top:1px solid var(--line);padding:3px 0;}.row:first-of-type{border-top:none;}.row b{font-variant-numeric:tabular-nums;color:var(--tx);}
  .cake div{font-size:10.5px;color:var(--tx3);padding:2px 6px;border-radius:5px;margin:0 0 1px;border-left:2px solid transparent;}
  .cake div.hot{color:var(--tx);background:rgba(255,255,255,.05);border-left-color:#38bdf8;}
  .net{font-size:11px;color:var(--tx2);line-height:1.5;margin:2px 0;}
  .lab{position:fixed;font-size:10.5px;color:var(--tx2);pointer-events:none;text-shadow:0 1px 3px #000;}
  #sbox{top:70px;left:50%;transform:translateX(-50%);width:min(430px,calc(100vw - 700px));padding:12px 15px;display:none;z-index:30;box-shadow:0 10px 40px rgba(0,0,0,.5);}
  #sbox .nm{font-size:13px;font-weight:500;margin:0 0 6px;display:flex;align-items:center;gap:8px;}#sbox .nm .d{width:11px;height:11px;border-radius:3px;}
  #sbox .body{font-size:12px;line-height:1.55;color:var(--tx2);}#sbox .body b{color:var(--tx);}#sbox .x{position:absolute;top:8px;right:11px;cursor:pointer;color:var(--tx3);}
  #qa .q{font-size:11px;padding:4px 7px;margin:0 4px 4px 0;}#qa .ans{font-size:11.5px;color:var(--tx);line-height:1.5;border-top:1px solid var(--line);padding-top:7px;margin-top:5px;min-height:48px;}
  #qa .askrow{display:flex;gap:5px;margin:7px 0 0;}#qa .askrow input{flex:1;background:rgba(255,255,255,.05);border:.5px solid var(--line);border-radius:7px;color:var(--tx);padding:5px 8px;font:inherit;font-size:11.5px;}#qa .askrow button{padding:5px 11px;}
  .node{cursor:pointer;}.node.off{cursor:default;}
  ::-webkit-scrollbar{width:7px;}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:8px;}
  .hchip{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;border:.5px solid var(--line);color:var(--tx2);font-size:9px;line-height:1;margin-left:6px;cursor:pointer;vertical-align:middle;font-weight:700;}
  #sigpills{display:flex;gap:4px;margin:7px 0 0;}
  .sigpill{font-size:10.5px;padding:4px 9px;border-radius:7px;border:.5px solid var(--line);background:rgba(255,255,255,.03);color:var(--tx2);cursor:pointer;transition:.14s;}
  .sigpill:hover{color:var(--tx);}.sigpill.on{background:rgba(43,212,232,.16);color:var(--tx);border-color:var(--accent);}
  .hchip:hover{color:var(--tx);border-color:var(--accent);background:rgba(43,212,232,.14);}
  .hpop{position:fixed;z-index:400;width:258px;background:rgba(16,21,30,.985);border:.5px solid var(--line);border-radius:10px;padding:11px 13px;box-shadow:0 10px 34px rgba(0,0,0,.55);font-size:11.5px;color:var(--tx2);line-height:1.52;}
  .hpop b{color:var(--tx);}.hpop .hask{margin-top:10px;font-size:11px;padding:5px 11px;border-radius:7px;border:.5px solid var(--accent);background:rgba(43,212,232,.16);color:var(--tx);cursor:pointer;}.hpop .hask:hover{background:rgba(43,212,232,.28);}
  #hint{position:fixed;left:50%;bottom:14px;transform:translateX(-50%);z-index:25;pointer-events:none;max-width:calc(100vw - 28px);
        font-size:10.5px;letter-spacing:.2px;color:var(--tx2);background:rgba(18,24,34,.86);backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px);
        border:.5px solid var(--line);border-radius:999px;padding:5px 12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
        opacity:0;transition:opacity .35s ease;}
  #hint.show{opacity:1;}
  /* ---- narrow desktop: keep both columns, tighten them so nothing collides ---- */
  @media (max-width:980px){
    #colL,#ctl{width:272px;}#nav{width:176px;}
    #colR,#side,#qa{width:262px;}
    #top{max-width:calc(100vw - 580px);padding:7px 12px;}
    #top h1{font-size:13.5px;}#top .sub{font-size:10.5px;}
    #sbox{width:calc(100vw - 40px);}
  }
  /* ---- mobile: stop being a fixed-panel HUD. Cube becomes a hero, panels stack in flow,
         the page scrolls normally instead of four fixed panels piling on each other. ---- */
  @media (max-width:700px){
    html{height:auto;overflow:visible;}
    body{height:auto;min-height:100%;overflow-x:hidden;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:0 0 28px;}
    #c{position:static;display:block;width:100%;height:52vh;min-height:240px;}
    #top{position:static;transform:none;max-width:none;margin:10px;}
    .col{position:static;display:block;padding:0 10px;pointer-events:auto;}
    .col>.pan{position:static;margin:0 0 10px;}
    #colL,#colR{width:auto;}
    #nav,#ctl,#side,#qa{width:auto;max-width:none;max-height:none;overflow:visible;flex:none;}
    #bsvg{height:210px;}
    #hint{position:absolute;top:calc(52vh - 40px);bottom:auto;}
    #sbox{top:auto;bottom:12px;width:calc(100vw - 20px);z-index:60;}
  }
</style></head><body>
<canvas id="c"></canvas>
<div id="top" class="pan"><h1 id="ttl">OVN Atlas</h1><span class="sub" id="tsub">real engine output · drag to rotate · ⌘/Ctrl-scroll to zoom</span></div>
<div id="sbox" class="pan"><span class="x" id="sboxx">✕</span><div class="nm" id="sboxnm"></div><div class="body" id="sboxbody"></div></div>

<div id="colL" class="col">
<div id="nav" class="pan"><div class="h">BODY-MAP — drop a cube</div>
  <svg id="bsvg" viewBox="0 0 100 200"></svg>
  <div id="navhint" style="font-size:10px;color:var(--tx3);margin-top:4px;">click an organ to drop the cube in — all four tissues are live</div>
</div>

<div id="ctl" class="pan">
  <div class="grp"><span class="glbl">WHO LIVES HERE</span><div id="popbtns"></div></div>
  <div class="grp"><span class="glbl">GRADIENTS</span>
    <select id="grad"></select><span style="font-size:11px;color:var(--tx2);margin-left:6px;">depth</span>
    <input id="depth" type="range" min="0" max="100" value="50" style="width:96px;vertical-align:middle;">
    <div id="sigpills"><button class="sigpill" data-sig="sigma">diseased</button><button class="sigpill" data-sig="sigma_h">healthy</button><button class="sigpill" data-sig="sigma_delta">Δσ lesion</button></div>
    <div id="legend" style="display:none;margin:9px 0 2px;font-size:11px;color:var(--tx2);">
      <div id="lgbar" style="height:9px;border-radius:4px;border:.5px solid var(--line);"></div>
      <div style="display:flex;justify-content:space-between;margin-top:3px;font-variant-numeric:tabular-nums;"><span id="lglo"></span><span id="lgmid" style="color:var(--tx3);"></span><span id="lghi"></span></div>
      <div id="lgmeta" style="margin-top:5px;line-height:1.5;"></div>
    </div></div>
  <div class="grp"><span class="glbl">SURFACES — click a wall to read it</span><div id="surfbtns"></div></div>
  <div class="grp"><span class="glbl">STRUCTURE &amp; BODY LINK</span>
    <button class="btn on" data-k="labels"><span class="d" style="background:#cbd5e1"></span>labels</button>
    <button class="btn" data-k="connections"><span class="d" style="background:#94a3b8"></span>links</button>
    <button class="btn" data-k="omv"><span class="d" style="background:#fca5a5"></span>OMVs → body</button>
    <button class="btn" id="resetcam" style="border-style:dashed;">reset view</button></div>
  <div class="grp"><span class="glbl">TIME-PLAYBACK</span>
    <button class="btn" id="play"><span class="d" style="background:#f0d27a"></span><span id="playlbl">▶ play</span></button>
    <input id="frame" type="range" min="0" max="0" value="0" style="width:120px;vertical-align:middle;">
    <span id="ovl" style="display:none"><button class="btn on" data-k="cma"><span class="d" style="background:#f0d27a"></span>colony</button>
    <button class="btn on" data-k="mineral"><span class="d" style="background:#dbeafe"></span>mineral</button></span>
    <div id="phase" style="font-size:11px;color:var(--tx2);margin-top:6px;"></div></div>
</div>
</div>

<div id="colR" class="col">
<div id="side" class="pan">
  <div id="exp"></div>
  <div class="h4" style="margin-top:0">ψ — system state</div><div id="psi"></div>
  <div class="h4">Color key</div><div id="key"></div>
  <div class="h4">Network (GNN roles · binding)</div><div id="net"></div>
  <div class="cake"><div class="h4">8-layer cake (lit = in view)</div>
    <div id="ck1">1 · Topology</div><div id="ck2">2 · Terrain</div><div id="ck3">3 · Chemistry</div><div id="ck4">4 · Phages</div>
    <div id="ck5">5 · Defenses</div><div id="ck6">6 · Host immunity</div><div id="ck7">7 · OMV cascade</div><div id="ck8">8 · Exogenous</div></div>
</div>
<div id="qa" class="pan"><div class="glbl" id="qatitle">ASK ANYTHING — grounded in the live engine</div>
  <div id="qbtns"></div>
  <div class="askrow"><input id="askbox" placeholder="type your own question…" autocomplete="off"><button id="asksend">ask</button></div>
  <div class="ans" id="ans">Pick a question, or type your own.</div></div>
</div>
<div id="hint">drag to rotate · ⌘/Ctrl-scroll or pinch to zoom</div>
<div id="labels"></div>

/*__THREE__*/
<script>
/*__DATA__*/
(function(){
 var THREE=window.THREE; if(!THREE){document.getElementById('exp').innerHTML='3-D engine failed to load (needs internet the first time it was generated).';return;}
 // built-in ORAL surfaces (the proven defaults); artery overrides via DATA.surfaces
 var ORAL_SURF={
  enamel:{name:'Tooth · enamel + pellicle',col:'#f5d9c0',rough:0.32,metal:0.06,html:'<b>Native surface:</b> enamel coated by the <b>acquired pellicle</b> (salivary proteins).<br><b>Information:</b> the <b>adhesion API</b> — which proteins decide who docks first.<br><b>Exchange:</b> Ca²⁺/PO₄³⁻ — biomineralization (calculus).'},
  gum:{name:'Gum · soft tissue',col:'#fb7185',rough:0.72,metal:0,html:'<b>Native surface:</b> pocket epithelium + blood/immune cells.<br><b>Information:</b> the <b>host–immune interface</b>.<br><b>Transports — the systemic gateway:</b> bacteremia + <b>OMVs</b> cross into blood; heme flows out when it bleeds.'},
  oral:{name:'Open mouth',col:'#7dd3fc',rough:0.08,metal:0,html:'<b>Native surface:</b> the saliva-bathed opening.<br><b>Exchange:</b> O₂, nitrate (→ NO), sugars; clears acids.'},
  deep:{name:'Deep pocket',col:'#64748b',rough:0.95,metal:0,html:'<b>Native surface:</b> the anoxic refuge — red-complex keystones accumulate; lowest O₂, highest heme.'},
  biofilm:{name:'Adjacent biofilm',col:'#94a3b8',rough:0.92,metal:0,html:'<b>Native surface:</b> the surrounding biofilm — a periodic unit-cell boundary.'}};
 var EXPL={oral:'<b>What am I looking at?</b> A cube of dental-plaque biofilm ~1mm across, in the pocket between tooth and gum. <b>Top</b>=open mouth (O₂); <b>bottom</b>=deep pocket (airless). One wall is tooth, the opposite is gum. Each dot is a patch of bacteria placed where the engine says it survives.',
  artery:'<b>What am I looking at?</b> A cube of artery wall. <b>Top</b>=the blood lumen (where the oral OMV load arrives); below it the intima where <b>foam cells</b> form the lipid core; deep=the muscular media. Each dot is a cell placed by the engine. Watch endothelial activation rise as you play time.',
  brain:'<b>What am I looking at?</b> A cube of brain tissue (the neurovascular unit). <b>Top</b>=the blood capillary + <b>BBB</b> (where the oral inflammatory load arrives); below it astrocytes and neurons in the parenchyma. <b>Microglia</b> activate and the <b>glymphatic</b> system clears amyloid — unless inflammation suppresses it. Play time to watch BBB integrity fall and amyloid build (the periodontitis → Alzheimer\'s route).',
  liver:'<b>What am I looking at?</b> A cube of liver tissue (a hepatic lobule). <b>Top</b>=the blood sinusoid — the liver gets <b>first-pass portal blood</b>, so translocating oral/gut bacteria and OMVs arrive here. Below it: hepatocytes (detox/metabolism), Kupffer macrophages, and <b>stellate cells</b>. The arriving load activates Kupffer + stellate cells → <b>fibrosis</b>, while metabolic stress drives <b>steatosis</b> (fat). Play time to watch fat and fibrosis build (the periodontitis → NAFLD/NASH route).'};

 var cv=document.getElementById('c');
 var renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});renderer.setPixelRatio(Math.min(2,devicePixelRatio||1));renderer.setClearColor(0x05070b,1);
 var scene=new THREE.Scene();var cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.1,1000);
 var dist0=26,dist=26,yaw0=0.7,pitch0=0.42,yaw=yaw0,pitch=pitch0;
 function upd(){cam.position.set(dist*Math.cos(pitch)*Math.sin(yaw),dist*Math.sin(pitch),dist*Math.cos(pitch)*Math.cos(yaw));cam.lookAt(0,0,0);}
 function vwid(){return Math.max(1,cv.clientWidth||innerWidth);}
 function vhgt(){return Math.max(1,cv.clientHeight||innerHeight);}
 function size(){var w=vwid(),h=vhgt();renderer.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();if(composer)composer.setSize(w,h);}
 scene.add(new THREE.AmbientLight(0x5a6b86,0.95));var dir=new THREE.DirectionalLight(0xfff4e6,0.85);dir.position.set(9,15,11);scene.add(dir);
 var pl=new THREE.PointLight(0x86b4ff,0.5,90);pl.position.set(-13,5,9);scene.add(pl);scene.fog=new THREE.FogExp2(0x06090e,0.0085);
 (function(){var c=document.createElement('canvas');c.width=2;c.height=256;var x=c.getContext('2d');var g=x.createLinearGradient(0,0,0,256);g.addColorStop(0,'#0e1421');g.addColorStop(1,'#05070b');x.fillStyle=g;x.fillRect(0,0,2,256);scene.background=new THREE.CanvasTexture(c);})();
 var GLOW=(function(){var c=document.createElement('canvas');c.width=c.height=64;var g=c.getContext('2d');var rg=g.createRadialGradient(32,32,0,32,32,32);rg.addColorStop(0,'rgba(255,255,255,1)');rg.addColorStop(0.28,'rgba(255,255,255,0.82)');rg.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=rg;g.fillRect(0,0,64,64);return new THREE.CanvasTexture(c);})();
 function glowMat(color,sz){return new THREE.PointsMaterial({color:new THREE.Color(color),size:sz,map:GLOW,transparent:true,opacity:0.9,depthWrite:false,blending:THREE.NormalBlending});}
 var composer=null;if(THREE.EffectComposer&&THREE.UnrealBloomPass&&THREE.RenderPass){composer=new THREE.EffectComposer(renderer);composer.addPass(new THREE.RenderPass(scene,cam));composer.addPass(new THREE.UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight),0.18,0.3,0.55));}
 var S=10;function world(px,py,pz){return [(px-0.5)*S,(pz-0.5)*S,(py-0.5)*S];}
 var grp=new THREE.Group();scene.add(grp);

 // ---- per-tissue state ----
 var D=null,objs={},state={},FR=[],curFrame=0,demoKind='matruchotii',SURF=ORAL_SURF,surfOn={},shownSurf=null,wpos=[];
 var faceMeshes={};
 function clearGroup(){while(grp.children.length){var o=grp.children[0];grp.remove(o);}objs={};faceMeshes={};}

 function buildFaces(){
  grp.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(S,S,S)),new THREE.LineBasicMaterial({color:0x33414f})));
  var defs=[['enamel',0,Math.PI/2,-S/2,0,0],['gum',0,Math.PI/2,S/2,0,0],['oral',Math.PI/2,0,0,S/2,0],['deep',Math.PI/2,0,0,-S/2,0],['biofilm',0,0,0,0,-S/2],['biofilm',0,0,0,0,S/2]];
  defs.forEach(function(d){var k=d[0],p=SURF[k];var m=new THREE.Mesh(new THREE.PlaneGeometry(S,S),new THREE.MeshStandardMaterial({color:new THREE.Color(p.col),roughness:p.rough,metalness:p.metal,transparent:true,opacity:0.1,side:THREE.DoubleSide,depthWrite:false}));m.rotation.x=d[1];m.rotation.y=d[2];m.position.set(d[3],d[4],d[5]);grp.add(m);(faceMeshes[k]=faceMeshes[k]||[]).push(m);});
 }
 function setSurface(key,on){surfOn[key]=on;(faceMeshes[key]||[]).forEach(function(m){m.material.opacity=on?0.9:0.1;m.material.depthWrite=!!on;});var b=document.querySelector('.btn[data-s="'+key+'"]');if(b)b.classList.toggle('on',!!on);
  if(on){var inf=SURF[key];document.getElementById('sboxnm').innerHTML='<span class="d" style="background:'+inf.col+'"></span>'+inf.name;document.getElementById('sboxbody').innerHTML=inf.html;document.getElementById('sbox').style.display='block';shownSurf=key;}
  else if(shownSurf===key){document.getElementById('sbox').style.display='none';shownSurf=null;}}

 function buildPoints(points){
  wpos=points.map(function(p){return world(p[0],p[1],p[2]);});
  D.groups.forEach(function(g){var pos=[];for(var i=0;i<points.length;i++){if(D.groups[points[i][3]]===g)pos.push(wpos[i][0],wpos[i][1],wpos[i][2]);}
   var geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
   var pt=new THREE.Points(geo,glowMat(D.group_color[g],g==='fungi'?0.6:0.44));pt.visible=!!state[g];grp.add(pt);objs[g]=pt;});
 }
 function buildEdges(){var ep=[];(D.edges||[]).forEach(function(e){var a=wpos[e[0]],b=wpos[e[1]];if(a&&b)ep.push(a[0],a[1],a[2],b[0],b[1],b[2]);});
  var eg=new THREE.BufferGeometry();eg.setAttribute('position',new THREE.Float32BufferAttribute(ep,3));objs.connections=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:0x94a3b8,transparent:true,opacity:0.22}));objs.connections.visible=!!state.connections;grp.add(objs.connections);}
 var omvBase=[],omvPh=[],og=null;
 function buildOMV(){omvBase=(D.omv||[]).map(function(i){return wpos[i];}).filter(Boolean);var pos=[];omvPh=[];omvBase.forEach(function(b){pos.push(b[0],b[1],b[2]);omvPh.push(Math.random());});og=new THREE.BufferGeometry();og.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));objs.omv=new THREE.Points(og,glowMat(0xfca5a5,0.42));objs.omv.visible=!!state.omv;grp.add(objs.omv);}

 // gradient slice
 var fcanvas=document.createElement('canvas'),fctx=null,ftex=null,slice=null;
 function buildGradient(){var fn=D.field_n;fcanvas.width=fcanvas.height=fn;fctx=fcanvas.getContext('2d');ftex=new THREE.CanvasTexture(fcanvas);ftex.minFilter=THREE.LinearFilter;ftex.magFilter=THREE.LinearFilter;
  slice=new THREE.Mesh(new THREE.PlaneGeometry(S,S),new THREE.MeshBasicMaterial({map:ftex,transparent:true,opacity:0.82,side:THREE.DoubleSide,depthWrite:false}));slice.rotation.x=-Math.PI/2;slice.visible=false;grp.add(slice);}
 function L(a,b,t){return Math.round(a+(b-a)*t);}
 function ramp(n,v){if(n==='O2')return[L(8,70,v),L(22,205,v),L(55,235,v)];if(n==='pH')return v<0.5?[L(232,205,v/0.5),L(70,180,v/0.5),L(60,95,v/0.5)]:[L(205,70,(v-0.5)/0.5),L(180,150,(v-0.5)/0.5),L(95,232,(v-0.5)/0.5)];
  if(n==='NO')return[L(12,60,v),L(30,232,v),L(25,140,v)];if(n==='heme')return[L(22,205,v),L(8,30,v),L(15,45,v)];
  if(n==='oxLDL')return[L(30,240,v),L(34,150,v),L(40,40,v)];if(n==='inflammation')return[L(24,245,v),L(34,70,v),L(44,55,v)];
  if(n==='amyloid')return[L(34,214,v),L(30,70,v),L(54,205,v)];if(n==='glymph')return[L(18,60,v),L(34,205,v),L(46,190,v)];
  if(n==='ecm')return[L(28,210,v),L(32,170,v),L(40,110,v)];if(n==='lipid')return[L(34,242,v),L(34,206,v),L(42,72,v)];
  if(n==='sigma'||n==='sigma_h')return[L(10,255,Math.min(1,v*1.4)),L(4,225,v*v),L(35,80,v)];
  if(n==='sigma_delta'){var t=(v-0.5)*2;if(t>=0)return[L(22,235,t),L(26,86,t),L(34,60,t)];return[L(22,70,-t),L(28,150,-t),L(36,235,-t)];}
  return[128,128,128];}
 function fmtN(x){if(!isFinite(x))return'–';if(x===0)return'0';var a=Math.abs(x);if(a>=1e4||a<1e-2){var e=Math.floor(Math.log10(a));return(x/Math.pow(10,e)).toFixed(1)+'e'+e;}return(Math.round(x*100)/100).toString();}
 function drawLegend(n){var lg=document.getElementById('legend');if(n==='off'||!D.fields[n]){lg.style.display='none';return;}lg.style.display='block';
  var st=[];for(var i=0;i<=12;i++){var c=ramp(n,i/12);st.push('rgb('+c[0]+','+c[1]+','+c[2]+')');}
  document.getElementById('lgbar').style.background='linear-gradient(90deg,'+st.join(',')+')';
  var f=D.fields[n],gf=(D.grad_fields.filter(function(g){return g.k===n;})[0]||{}),lab=gf.label||n;
  var EL=function(id){return document.getElementById(id);};
  if((n==='sigma'||n==='sigma_h')&&D.sigma_meta){var sm=(n==='sigma_h'?(D.sigma_meta_h||D.sigma_meta):D.sigma_meta),state=(n==='sigma_h'?'healthy':'diseased');
   EL('lglo').textContent='0';EL('lghi').textContent=fmtN(sm.max);EL('lgmid').textContent='σ density · '+state;
   var drv=(sm.by_species||[]).map(function(s){return s[0];}).join(' · ')||'—';
   EL('lgmeta').innerHTML='<b style="color:var(--tx)">Σ total '+fmtN(sm.total)+'</b> · driven by <b style="color:var(--tx)">'+drv+'</b><br><span style="color:var(--tx3)">irreversible free-energy dissipation; brighter = more ('+state+' state · log colour).<br>Units: '+(sm.units||'relative, not calibrated to W/K')+'.</span>';
  }else if(n==='sigma_delta'&&D.sigma_delta_meta){var dm=D.sigma_delta_meta;
   EL('lglo').textContent='− removed';EL('lghi').textContent='added +';EL('lgmid').textContent='Δσ lesion';
   var drv2=(dm.by_species||[]).map(function(s){return s[0];}).join(' · ')||'—';
   EL('lgmeta').innerHTML='<b style="color:var(--tx)">lesion adds '+fmtN(dm.added)+' · removes '+fmtN(dm.removed)+'</b><br>carried by <b style="color:var(--tx)">'+drv2+'</b><br><span style="color:var(--tx3)">disease − health: the shared O₂ supply cancels, leaving the lesion. <span style="color:#e8895f">red = disease ADDS</span>, <span style="color:#5b9be8">blue = removes</span>.</span>';
  }else{
   EL('lglo').textContent=fmtN(f.min);EL('lghi').textContent=fmtN(f.max);EL('lgmid').textContent=lab;
   EL('lgmeta').innerHTML='<span style="color:var(--tx3)">slice value range (model units)</span>';
  }}
 var sigVar='sigma';   // which σ view the pills select; the dropdown carries a single "σ" entry
 function syncPills(ef){var on=(ef==='sigma'||ef==='sigma_h'||ef==='sigma_delta');var ps=document.querySelectorAll('.sigpill');for(var i=0;i<ps.length;i++)ps[i].classList.toggle('on',on&&ps[i].getAttribute('data-sig')===ef);}
 function drawSlice(){var n=document.getElementById('grad').value,ef=(n==='sigma'?sigVar:n),d=document.getElementById('depth').value/100;drawLegend(ef);syncPills(ef);
  if(n==='off'||!D.fields[ef]){slice.visible=false;return;}slice.visible=true;slice.position.y=(d-0.5)*S;var fn=D.field_n,q=D.fields[ef].q,k=Math.round(d*(fn-1)),img=fctx.createImageData(fn,fn);
  for(var v=0;v<fn;v++)for(var u=0;u<fn;u++){var val=q[(u*fn+v)*fn+k]/255,c=ramp(ef,val),o=(v*fn+u)*4;img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=235;}fctx.putImageData(img,0,0);ftex.needsUpdate=true;}

 // time-playback (matruchotii overlay OR progression cloud)
 function ptsObj(arr,color,sz){var pos=[];arr.forEach(function(p){var w=world(p[0],p[1],p[2]);pos.push(w[0],w[1],w[2]);});var g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));return new THREE.Points(g,glowMat(color,sz));}
 function loadFrame(i){curFrame=i;var f=FR[i];if(!f)return;
  if(demoKind==='matruchotii'){grp.remove(objs.cma);objs.cma=ptsObj(f.cma,D.demo.cma_color,0.45);objs.cma.visible=!!state.cma;grp.add(objs.cma);
   grp.remove(objs.mineral);objs.mineral=ptsObj(f.mineral,D.demo.mineral_color,0.52);objs.mineral.visible=!!state.mineral;grp.add(objs.mineral);
   document.getElementById('phase').innerHTML=(f.min>1?'<b style="color:#dbeafe">Phase 2 · biomineralization</b>':'<b style="color:#f0d27a">Phase 1 · colonization</b>')+' · t='+f.t+'h · colony '+f.cma_n+' · mineral '+f.min;}
  else{ // progression: swap the whole cell cloud
   D.groups.forEach(function(g){if(objs[g]){grp.remove(objs[g]);}});
   wpos=f.pts.map(function(p){return world(p[0],p[1],p[2]);});
   D.groups.forEach(function(g){var pos=[];for(var j=0;j<f.pts.length;j++){if(D.groups[f.pts[j][3]]===g)pos.push(wpos[j][0],wpos[j][1],wpos[j][2]);}var geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));var pt=new THREE.Points(geo,glowMat(D.group_color[g],0.46));pt.visible=!!state[g];grp.add(pt);objs[g]=pt;});
   document.getElementById('phase').innerHTML='step '+(curFrame+1)+'/'+FR.length+' · model time t='+f.t+' (arbitrary units — no calibrated clock)'+(f.info?' · '+f.info:'');}}
 var playing=false,ptimer=null;
 function setPlay(p){playing=p;document.getElementById('playlbl').textContent=p?'❚❚ pause':'▶ play';document.getElementById('play').classList.toggle('on',p);
  if(p){ptimer=setInterval(function(){var n=curFrame+1;if(n>=FR.length)n=0;document.getElementById('frame').value=n;loadFrame(n);},260);}else clearInterval(ptimer);}

 // labels
 var LBLS=[],labWrap=document.getElementById('labels');
 function buildLabels(){labWrap.innerHTML='';LBLS=[['enamel',-S/2-0.4,0,0],['gum',S/2+0.4,0,0],['oral',0,S/2+0.4,0],['deep',0,-S/2-0.4,0],['biofilm',0,0,S/2+0.4]].map(function(o){var e=document.createElement('div');e.className='lab';e.textContent=SURF[o[0]].name;labWrap.appendChild(e);return{el:e,p:new THREE.Vector3(o[1],o[2],o[3])};});}
 function overPanel(x,y){var ids=['top','nav','ctl','side','qa','sbox'];for(var i=0;i<ids.length;i++){var el=document.getElementById(ids[i]);if(!el||el.style.display==='none')continue;var r=el.getBoundingClientRect();if(x>=r.left&&x<=r.right&&y>=r.top&&y<=r.bottom)return true;}return false;}
 function updLabels(){var on=state.labels;if(!on){LBLS.forEach(function(o){o.el.style.display='none';});return;}
  var r=cv.getBoundingClientRect();
  LBLS.forEach(function(o){var v=o.p.clone().project(cam),px=r.left+(v.x*0.5+0.5)*r.width,py=r.top+(-v.y*0.5+0.5)*r.height;
   if(v.z>1||px<r.left+4||px>r.right-4||py<r.top+4||py>r.bottom-4||overPanel(px,py)){o.el.style.display='none';return;}
   o.el.style.display='block';o.el.style.left=px+'px';o.el.style.top=py+'px';o.el.style.transform='translate(-50%,-50%)';});}

 function applyState(){D.groups.forEach(function(g){if(objs[g])objs[g].visible=!!state[g];});if(objs.connections)objs.connections.visible=!!state.connections;if(objs.omv)objs.omv.visible=!!state.omv;if(objs.cma)objs.cma.visible=!!state.cma;if(objs.mineral)objs.mineral.visible=!!state.mineral;updLabels();}

 // ---- panels ----
 function fillPanels(){
  document.getElementById('exp').innerHTML=(EXPL[D.name]||'')+'<div style="margin-top:8px;padding-top:7px;border-top:1px solid var(--line);color:var(--tx3);font-size:10.5px;">New here? Tap any <b style="color:var(--accent)">?</b> for a plain-English explainer — or ask the <b>answer man</b> (Claude) anything below ↓</div>';
  // ψ
  var ph='';for(var k in D.psi){ph+='<div class="row"><span>'+k+'</span><b>'+D.psi[k]+'</b></div>';}document.getElementById('psi').innerHTML=ph;
  // key + pop buttons
  var pb=document.getElementById('popbtns');pb.innerHTML='';var ky=document.getElementById('key');ky.innerHTML='';
  D.groups.forEach(function(g){var lab=(D.group_label&&D.group_label[g])||g;
   var b=document.createElement('button');b.className='btn on';b.setAttribute('data-k',g);b.innerHTML='<span class="d" style="background:'+D.group_color[g]+'"></span>'+lab;pb.appendChild(b);
   var kk=document.createElement('div');kk.className='key';kk.innerHTML='<span class="d" style="background:'+D.group_color[g]+'"></span>'+lab;ky.appendChild(kk);});
  // surfaces buttons
  var sb=document.getElementById('surfbtns');sb.innerHTML='';['enamel','gum','oral','deep','biofilm'].forEach(function(k){var b=document.createElement('button');b.className='btn';b.setAttribute('data-s',k);b.innerHTML='<span class="d" style="background:'+SURF[k].col+'"></span>'+SURF[k].name;sb.appendChild(b);});
  // gradient options
  var gs=document.getElementById('grad');gs.innerHTML='<option value="off">— gradient: off —</option>';D.grad_fields.forEach(function(f){if(f.k==='sigma_h'||f.k==='sigma_delta')return;var o=document.createElement('option');o.value=f.k;o.textContent=(f.k==='sigma'?'entropy production σ':f.label);gs.appendChild(o);});
  // network panel
  var net=document.getElementById('net'),ai=D.ai||{};var nh='';
  if(ai.coagg&&ai.coagg.length){nh+='<div class="net">Top realised coaggregations (co-presence × GCN affinity), with molecular basis:</div>';
   ai.coagg.slice(0,5).forEach(function(e){var bnd=e.binding,mol=bnd.basis==='curated structure'?(bnd.adhesin+' ↔ '+bnd.receptor):('affinity '+bnd.score+', '+bnd.basis);nh+='<div class="net">• <b>'+e.a+'–'+e.b+'</b> (w '+e.weight+') · '+mol+'</div>';});
   nh+='<div class="net" style="color:var(--tx3)">GCN R²≈'+ai.r2+' (weak, real) · binding curated / ESMFold-pending</div>';}
  else nh='<div class="net" style="color:var(--tx3)">No microbial coaggregation network in this tissue.</div>';
  net.innerHTML=nh;
  // cake
  for(var i=1;i<=8;i++){var e=document.getElementById('ck'+i);if(e)e.classList.toggle('hot',(D.cake_hot||[]).indexOf(i)>=0);}
  // Q&A
  var qb=document.getElementById('qbtns');qb.innerHTML='';(D.qa||[]).forEach(function(x,i){var b=document.createElement('button');b.className='btn q';b.textContent=x.q;b.onclick=function(){document.getElementById('ans').textContent=x.a;};qb.appendChild(b);});
  document.getElementById('ans').textContent='Pick a question.';
  // wire dynamic buttons
  document.querySelectorAll('#popbtns .btn').forEach(function(b){if(b.__w)return;b.__w=1;b.addEventListener('click',function(){var k=b.getAttribute('data-k');if(k==null)return;state[k]=state[k]?0:1;b.classList.toggle('on',!!state[k]);applyState();});});
  document.querySelectorAll('.btn[data-s]').forEach(function(b){b.addEventListener('click',function(){var k=b.getAttribute('data-s');setSurface(k,!surfOn[k]);});});
 }

 function loadTissue(name){
  D=ATLAS.tissues[name];SURF=D.surfaces||ORAL_SURF;demoKind=D.demo_kind||'matruchotii';surfOn={};shownSurf=null;
  document.getElementById('sbox').style.display='none';
  document.getElementById('ttl').textContent='OVN Atlas — '+D.tissue;
  // reset state
  state={};D.groups.forEach(function(g){state[g]=1;});state.labels=1;state.connections=0;state.omv=0;state.cma=1;state.mineral=1;
  clearGroup();buildFaces();
  var basePts=(demoKind==='progression'&&D.demo.frames.length)?D.demo.frames[D.demo.frames.length-1].pts:D.points;
  buildPoints(basePts);buildEdges();buildOMV();buildGradient();buildLabels();
  // playback frames
  if(demoKind==='matruchotii'){FR=D.demo.frames;document.getElementById('ovl').style.display='inline';objs.cma=ptsObj([],D.demo.cma_color,0.45);grp.add(objs.cma);objs.mineral=ptsObj([],D.demo.mineral_color,0.52);grp.add(objs.mineral);}
  else{FR=D.demo.frames;document.getElementById('ovl').style.display='none';}
  document.getElementById('frame').max=Math.max(0,FR.length-1);document.getElementById('frame').value=0;
  fillPanels();applyState();drawSlice();loadFrame(demoKind==='matruchotii'?0:FR.length-1);
  // reflect active node
  document.querySelectorAll('.node').forEach(function(n){n.setAttribute('stroke-width',n.getAttribute('data-id')===name?'3':'1');});
 }

 // ---- body-map SVG ----
 (function(){var svg=document.getElementById('bsvg');var NS='http://www.w3.org/2000/svg';
  function el(t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);return e;}
  // simple body silhouette
  svg.appendChild(el('ellipse',{cx:50,cy:18,rx:11,ry:13,fill:'rgba(255,255,255,.05)',stroke:'#33414f'}));
  svg.appendChild(el('path',{d:'M50 31 C34 33 30 48 31 86 L34 132 L42 132 L44 96 L56 96 L58 132 L66 132 L69 86 C70 48 66 33 50 31 Z',fill:'rgba(255,255,255,.05)',stroke:'#33414f'}));
  svg.appendChild(el('path',{d:'M33 50 L20 88 M67 50 L80 88',stroke:'#33414f','stroke-width':5,fill:'none','stroke-linecap':'round',opacity:0.5}));
  // links
  ATLAS.bodymap.links.forEach(function(lk){var a=ATLAS.bodymap.nodes.filter(function(n){return n.id===lk.from;})[0],b=ATLAS.bodymap.nodes.filter(function(n){return n.id===lk.to;})[0];if(a&&b)svg.appendChild(el('line',{x1:a.x*100,y1:a.y*200,x2:b.x*100,y2:b.y*200,stroke:'#c0392b','stroke-width':1.4,'stroke-dasharray':'2 2',opacity:0.8}));});
  ATLAS.bodymap.nodes.forEach(function(n){var c=el('circle',{cx:n.x*100,cy:n.y*200,r:n.active?4.5:3.4,fill:n.active?'#2bd4e8':'#3a4654',stroke:'#cfe2ee','stroke-width':1,'data-id':n.id,class:'node'+(n.active?'':' off')});
   if(n.active){c.addEventListener('click',function(){loadTissue(n.id);});c.addEventListener('mouseenter',function(){document.getElementById('navhint').textContent=n.label;});}
   svg.appendChild(c);
   var t=el('text',{x:n.x*100+6,y:n.y*200+3,fill:n.active?'#cfe2ee':'#5f6f80','font-size':'6'});t.textContent=n.id;svg.appendChild(t);});
 })();

 // static controls
 document.getElementById('grad').addEventListener('change',drawSlice);
 document.querySelectorAll('.sigpill').forEach(function(p){p.addEventListener('click',function(){sigVar=p.getAttribute('data-sig');var g=document.getElementById('grad');g.value='sigma';g.dispatchEvent(new Event('change'));});});
 document.getElementById('depth').addEventListener('input',drawSlice);
 document.getElementById('resetcam').addEventListener('click',function(){dist=dist0;yaw=yaw0;pitch=pitch0;upd();});
 document.getElementById('frame').addEventListener('input',function(){if(playing)setPlay(false);loadFrame(+this.value);});
 document.getElementById('play').addEventListener('click',function(){setPlay(!playing);});
 document.getElementById('sboxx').addEventListener('click',function(){document.getElementById('sbox').style.display='none';shownSurf=null;});
 // static structure buttons (labels/connections/omv/cma/mineral) — wire once
 document.querySelectorAll('#ctl .btn[data-k]').forEach(function(b){b.addEventListener('click',function(){var k=b.getAttribute('data-k');state[k]=state[k]?0:1;b.classList.toggle('on',!!state[k]);applyState();});});

 // live ask-anything — POSTs the current tissue's REAL digest to the backend; Claude grounds on it.
 function liveAsk(q){if(!q)return;var ans=document.getElementById('ans');ans.textContent='… asking Claude over the live engine state …';
  var url=window.OVN_ASK_URL||'/api/ask';
  fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,tissue:D.name,tissueLabel:D.tissue,digest:D.digest||{}})})
   .then(function(r){if(!r.ok)throw 0;return r.json();})
   .then(function(d){ans.textContent=d.answer||'(no answer)';})
   .catch(function(){ans.textContent='The live assistant needs the deployed backend (not reachable here). The preset questions above still answer directly from the live engine.';});}
 document.getElementById('asksend').addEventListener('click',function(){liveAsk(document.getElementById('askbox').value.trim());});
 document.getElementById('askbox').addEventListener('keydown',function(e){if(e.key==='Enter')liveAsk(this.value.trim());});

 // ---- newcomer context help: a "?" on every section that explains it + asks the answer man ----
 var HELP={
  'WHO LIVES':{x:"Each glowing dot is a patch of one microbe type, dropped where the engine computes it can survive. Colours group them by job — protective nitrate-reducers, bridge connectors, aggressive red-complex pathogens.",q:"What cells live in this cube and what is each one doing?"},
  'GRADIENTS':{x:"Lay a heat-map slice through the cube for one quantity — oxygen, acidity, or <b>σ</b> (entropy production: where the tissue is burning free energy). <b>Δσ</b> shows the lesion only — diseased minus healthy, so the oxygen background cancels out.",q:"What is the σ entropy-production gradient showing me?"},
  'SURFACES':{x:"The six walls are real interfaces — tooth enamel, gum, open mouth, the bloodstream. Click a wall to read what crosses it.",q:"What are the surfaces of this cube and what passes through them?"},
  'BODY LINK':{x:"The thin lines are who-sticks-to-whom (coaggregation). <b>OMVs</b> are nanoparticles of bacterial toxin that bud off and ride the bloodstream to distant organs — the oral→body route.",q:"What are OMVs and how do they reach the rest of the body?"},
  'TIME-PLAYBACK':{x:"Press play to run the engine forward and watch the disease develop — first colonisation, then the lesion building.",q:"What changes over time as this disease develops?"},
  'BODY-MAP':{x:"Click an organ to drop the cube into that tissue — mouth, artery, brain, or liver. The very same engine runs in each; only the cells and chemistry change.",q:"How does one model run in the mouth, heart, brain and liver?"},
  'SYSTEM STATE':{x:"<b>ψ</b> (psi) is the cube's state — the engine's own readout (dysbiosis, OMV flux, BBB integrity, fibrosis). These are model predictions, not lab measurements.",q:"What do the ψ system-state numbers mean here?"},
  'ASK ANYTHING':{x:"This is the <b>answer man</b> — Claude, reading the live engine state of THIS cube. Ask anything in plain English; it answers from the actual numbers, not a script.",q:"Give me a plain-English summary of what's happening in this cube."}
 };
 function closeHelp(){var p=document.getElementById('hpop');if(p)p.remove();document.removeEventListener('click',closeHelp);}
 function showHelp(a,h){closeHelp();var pop=document.createElement('div');pop.className='hpop';pop.id='hpop';
  pop.innerHTML='<div>'+h.x+'</div><button class="hask">Ask the answer man →</button>';document.body.appendChild(pop);
  var r=a.getBoundingClientRect();pop.style.left=Math.max(8,Math.min(r.left-8,innerWidth-272))+'px';pop.style.top=(r.bottom+7)+'px';
  pop.querySelector('.hask').onclick=function(ev){ev.stopPropagation();closeHelp();var qa=document.getElementById('qa');if(qa)qa.scrollIntoView({behavior:'smooth',block:'nearest'});var ab=document.getElementById('askbox');if(ab)ab.value=h.q;liveAsk(h.q);};
  setTimeout(function(){document.addEventListener('click',closeHelp);},20);}
 function buildHelpChips(){document.querySelectorAll('.glbl, #nav .h, #side .h4, #qatitle').forEach(function(g){
   if(g.querySelector('.hchip'))return;var t=g.textContent.toUpperCase(),h=null;for(var k in HELP){if(t.indexOf(k)>-1){h=HELP[k];break;}}if(!h)return;
   var c=document.createElement('span');c.className='hchip';c.textContent='?';c.title='what is this?';c.onclick=function(e){e.stopPropagation();showHelp(c,h);};g.appendChild(c);});}

 loadTissue('oral');buildHelpChips();upd();size();

 var drag=false,lx=0,ly=0,PTR={},pinD=0,pinZ=0,pmx=0,pmy=0,engaged=false,engT=0;
 var HINT_DEF='drag to rotate · ⌘/Ctrl-scroll or pinch to zoom',hintEl=document.getElementById('hint'),hintT=null;
 function flashHint(msg,ms){if(!hintEl)return;hintEl.textContent=msg||HINT_DEF;hintEl.classList.add('show');clearTimeout(hintT);hintT=setTimeout(function(){hintEl.classList.remove('show');},ms||2800);}
 function engage(){engaged=true;engT=Date.now();}
 function armed(){return engaged&&(Date.now()-engT)<4000;}
 function plist(){var a=[],k;for(k in PTR)a.push(PTR[k]);return a;}
 function setDist(d){dist=Math.max(13,Math.min(48,d));upd();}
 cv.addEventListener('pointerdown',function(e){PTR[e.pointerId]={x:e.clientX,y:e.clientY};engage();
  var a=plist();if(a.length>=2){drag=false;var dx=a[0].x-a[1].x,dy=a[0].y-a[1].y;pinD=Math.sqrt(dx*dx+dy*dy);pinZ=dist;pmx=(a[0].x+a[1].x)/2;pmy=(a[0].y+a[1].y)/2;return;}
  drag=true;lx=e.clientX;ly=e.clientY;try{cv.setPointerCapture(e.pointerId);}catch(_){}});
 cv.addEventListener('pointermove',function(e){if(PTR[e.pointerId]){PTR[e.pointerId].x=e.clientX;PTR[e.pointerId].y=e.clientY;}
  var a=plist();
  if(a.length>=2){var dx=a[0].x-a[1].x,dy=a[0].y-a[1].y,d=Math.sqrt(dx*dx+dy*dy),mx=(a[0].x+a[1].x)/2,my=(a[0].y+a[1].y)/2;
   if(pinD>8&&d>8)dist=Math.max(13,Math.min(48,pinZ*pinD/d));
   yaw-=(mx-pmx)*0.006;pitch=Math.max(-1.35,Math.min(1.35,pitch+(my-pmy)*0.006));pmx=mx;pmy=my;engage();upd();return;}
  if(!drag)return;yaw-=(e.clientX-lx)*0.006;pitch=Math.max(-1.35,Math.min(1.35,pitch+(e.clientY-ly)*0.006));lx=e.clientX;ly=e.clientY;upd();});
 function pend(e){delete PTR[e.pointerId];pinD=0;var a=plist();if(!a.length){drag=false;}else{lx=a[0].x;ly=a[0].y;drag=true;}}
 cv.addEventListener('pointerup',pend);cv.addEventListener('pointercancel',pend);
 cv.addEventListener('pointerleave',function(){if(!plist().length)engaged=false;});
 // Wheel is a ZOOM only when the gesture is deliberate: ctrl/meta held (also how trackpad
 // pinch arrives), or the cube was touched in the last 4s. Otherwise we do NOT preventDefault,
 // so a plain scroll inside an iframe scrolls the host page instead of trapping the reader.
 cv.addEventListener('wheel',function(e){
  if(!(e.ctrlKey||e.metaKey)&&!armed()){flashHint('scroll is passing through to the page · hold ⌘/Ctrl, or click the cube first, to zoom');return;}
  e.preventDefault();var dy=e.deltaY*(e.deltaMode===1?16:(e.deltaMode===2?400:1));
  setDist(dist+(dy>0?1.6:-1.6));engage();},{passive:false});
 addEventListener('resize',size);addEventListener('orientationchange',function(){setTimeout(size,120);});
 flashHint(null,7000);
 var t=0;(function loop(){requestAnimationFrame(loop);if(!drag){yaw+=0.0006;upd();}
  if(objs.omv&&objs.omv.visible&&omvBase.length){t+=0.02;var a=og.attributes.position.array;for(var i=0;i<omvBase.length;i++){var ph=(omvPh[i]+t*0.4)%1;a[i*3]=omvBase[i][0]+ph*5.0;a[i*3+1]=omvBase[i][1]+ph*0.6;a[i*3+2]=omvBase[i][2];}og.attributes.position.needsUpdate=true;}
  if(composer)composer.render();else renderer.render(scene,cam);updLabels();})();
})();
</script></body></html>"""

