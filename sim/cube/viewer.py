"""Write a standalone HTML viewer with the REAL engine output baked in.

`build_sample` runs the actual cube (seed → snapshot solve → export); this module injects that
exact JSON into a self-contained Three.js page (Three.js inlined so the file works offline). No
JS-procedural approximation — what you see is the engine state.

    from sim.cube.viewer import write_viewer
    write_viewer("/Users/.../cube_viewer.html", dim=48)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from sim.cube.dynamics import build_demo

_THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
# three core + the post-processing chain for bloom (r128 examples/js attach to the THREE global).
_BUNDLE_URLS = [
    _THREE_URL,
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js",
]


def _three_bundle() -> str:
    """Inline three.js + the bloom post-processing chain. Every URL in ``_BUNDLE_URLS`` is
    MANDATORY: a partial bundle silently ships a bloom-less cube, and a CDN <script src> fallback
    silently ships a network-dependent one. Any failure aborts the build instead of degrading.

        html = _TEMPLATE.replace("/*__THREE__*/", _three_bundle())
    """
    out = []
    for u in _BUNDLE_URLS:
        try:
            with urllib.request.urlopen(u, timeout=25) as r:
                body = r.read().decode("utf-8", "ignore")
        except Exception as exc:
            raise RuntimeError(
                f"three.js bundle fetch FAILED for {u}: {exc!r}. All {len(_BUNDLE_URLS)} URLs are "
                "required for a self-contained cube - a partial bundle ships without bloom, and a "
                "CDN fallback ships a file that breaks offline. Refusing to write a degraded "
                "artifact. Fix the network, or vendor the file locally and point _BUNDLE_URLS at "
                "a file:// path, then rerun."
            ) from exc
        if len(body) < 256 or body.lstrip().startswith("<"):
            raise RuntimeError(
                f"three.js bundle fetch for {u} returned {len(body)} chars that do not look like "
                "JavaScript (empty body, or an HTML error page served with HTTP 200). Refusing to "
                "bake a truncated bundle into the cube."
            )
        out.append("<script>" + body + "</script>")
    return "\n".join(out)


def write_viewer(path: str, dim: int = 44, field_n: int = 20, n_points: int = 2200,
                 max_edges: int = 350, capture: int = 32) -> str:
    data = build_demo(dim=dim, field_n=field_n, n_points=n_points,
                      max_edges=max_edges, capture=capture)
    html = _TEMPLATE.replace("/*__THREE__*/", _three_bundle()) \
                    .replace("/*__DATA__*/", "const DATA=" + json.dumps(data) + ";")
    Path(path).write_text(html, encoding="utf-8")
    return path


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pocket biofilm — cube viewer</title>
<style>
  :root{--bg:#070a0f;--panel:rgba(18,24,34,.82);--line:rgba(255,255,255,.09);--tx:#e9eef5;--tx2:#9fb0c3;--tx3:#5f6f80;}
  *{box-sizing:border-box;} html,body{margin:0;height:100%;background:var(--bg);color:var(--tx);font-family:Inter,system-ui,Segoe UI,sans-serif;overflow:hidden;}
  #c{position:fixed;inset:0;}
  .pan{position:fixed;background:var(--panel);backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px);border:.5px solid var(--line);border-radius:14px;}
  #top{top:16px;left:16px;right:16px;padding:12px 16px;display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap;}
  #top h1{margin:0;font-size:17px;font-weight:500;letter-spacing:.2px;}
  #top .sub{font-size:12px;color:var(--tx2);}
  #ctl{left:16px;bottom:16px;width:330px;max-height:62vh;overflow:auto;padding:14px;}
  #side{right:16px;bottom:16px;width:300px;max-height:62vh;overflow:auto;padding:14px;}
  @media(max-width:820px){#ctl,#side{width:calc(50vw - 24px);}}
  .glbl{font-size:11px;font-weight:500;color:var(--tx2);text-transform:none;margin:2px 0 7px;display:block;letter-spacing:.3px;}
  .grp{margin:0 0 13px;}
  .btn{display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:6px 10px;margin:0 6px 6px 0;border-radius:8px;border:.5px solid var(--line);background:rgba(255,255,255,.03);color:var(--tx2);cursor:pointer;transition:.14s;}
  .btn .d{width:9px;height:9px;border-radius:50%;}
  .btn.on{background:rgba(255,255,255,.10);color:var(--tx);border-color:rgba(255,255,255,.28);}
  .btn:not(.on) .d{opacity:.32;}
  select,input[type=range]{font:inherit;color:var(--tx);} select{background:rgba(255,255,255,.05);border:.5px solid var(--line);border-radius:8px;padding:5px 8px;font-size:12px;}
  input[type=range]{accent-color:#2bd4e8;}
  #exp{font-size:12.5px;line-height:1.62;color:var(--tx2);background:rgba(255,255,255,.04);border:.5px solid var(--line);border-radius:10px;padding:11px 13px;margin:0 0 12px;}
  #exp b{color:var(--tx);font-weight:500;}
  .key{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--tx2);margin:0 0 5px;}
  .key .d{width:10px;height:10px;border-radius:50%;flex:none;}
  .h4{font-size:12px;font-weight:500;margin:0 0 8px;color:var(--tx);}
  .cake div{font-size:11px;color:var(--tx3);padding:3px 7px;border-radius:6px;margin:0 0 2px;border-left:2px solid transparent;}
  .cake div.hot{color:var(--tx);background:rgba(255,255,255,.05);border-left-color:#38bdf8;}
  .hint{position:fixed;left:50%;transform:translateX(-50%);bottom:8px;font-size:11px;color:var(--tx3);}
  .lab{position:fixed;font-size:10.5px;color:var(--tx2);pointer-events:none;text-shadow:0 1px 3px #000;letter-spacing:.3px;}
  #sbox{top:78px;left:50%;transform:translateX(-50%);width:min(460px,calc(100vw - 32px));padding:13px 16px;display:none;z-index:30;box-shadow:0 10px 40px rgba(0,0,0,.5);}
  #sbox .nm{font-size:13px;font-weight:500;margin:0 0 7px;display:flex;align-items:center;gap:8px;}
  #sbox .nm .d{width:11px;height:11px;border-radius:3px;}
  #sbox .body{font-size:12px;line-height:1.62;color:var(--tx2);}
  #sbox .body b{color:var(--tx);font-weight:500;}
  #sbox .x{position:absolute;top:9px;right:12px;cursor:pointer;color:var(--tx3);font-size:13px;}
  ::-webkit-scrollbar{width:8px;}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:8px;}
</style></head><body>
<canvas id="c"></canvas>
<div id="top" class="pan"><h1>Pocket biofilm — the cube</h1><span class="sub">real engine output · drag to rotate · scroll to zoom</span></div>
<div id="sbox" class="pan"><span class="x" id="sboxx">✕</span><div class="nm" id="sboxnm"></div><div class="body" id="sboxbody"></div></div>

<div id="ctl" class="pan">
  <div class="grp"><span class="glbl">WHO LIVES HERE — bacteria · fungi · viruses</span><div id="popbtns"></div></div>
  <div class="grp"><span class="glbl">CHEMICAL &amp; MOLECULAR GRADIENTS</span>
    <select id="grad"><option value="off">— gradient: off —</option><option value="O2">oxygen (O₂)</option><option value="pH">acidity (pH)</option><option value="NO">nitric oxide (NO)</option><option value="heme">heme (blood iron)</option></select>
    <span style="font-size:12px;color:var(--tx2);margin-left:8px;">depth</span>
    <input id="depth" type="range" min="0" max="100" value="50" style="width:120px;vertical-align:middle;">
  </div>
  <div class="grp"><span class="glbl">SURFACES — click a wall to make it opaque &amp; read what it is</span>
    <button class="btn" data-s="enamel"><span class="d" style="background:#f5d9c0"></span>tooth · enamel</button>
    <button class="btn" data-s="gum"><span class="d" style="background:#fb7185"></span>gum · tissue</button>
    <button class="btn" data-s="oral"><span class="d" style="background:#7dd3fc"></span>open mouth</button>
    <button class="btn" data-s="deep"><span class="d" style="background:#64748b"></span>deep pocket</button>
    <button class="btn" data-s="biofilm"><span class="d" style="background:#94a3b8"></span>adjacent biofilm</button>
  </div>
  <div class="grp"><span class="glbl">STRUCTURE &amp; THE BODY LINK</span>
    <button class="btn on" data-k="labels"><span class="d" style="background:#cbd5e1"></span>wall labels</button>
    <button class="btn" data-k="connections"><span class="d" style="background:#94a3b8"></span>coaggregation links</button>
    <button class="btn" data-k="omv"><span class="d" style="background:#fca5a5"></span>OMVs → the body</button>
    <button class="btn" id="resetcam" style="border-style:dashed;">reset view</button>
  </div>
  <div class="grp"><span class="glbl">TIME-PLAYBACK — <i>C. matruchotii</i> colonization → biomineralization</span>
    <button class="btn" id="play"><span class="d" style="background:#f0d27a"></span><span id="playlbl">▶ play</span></button>
    <input id="frame" type="range" min="0" max="0" value="0" style="width:148px;vertical-align:middle;">
    <button class="btn" data-k="cma"><span class="d" style="background:#f0d27a"></span>colony</button>
    <button class="btn" data-k="mineral"><span class="d" style="background:#dbeafe"></span>mineral</button>
    <div id="phase" style="font-size:11.5px;color:var(--tx2);margin-top:7px;line-height:1.5;"></div>
  </div>
</div>

<div id="side" class="pan">
  <div id="exp"><b>What am I looking at?</b> A cube of dental-plaque biofilm about a millimetre across, in the pocket between a tooth and the gum. <b>Top</b> = the open mouth (oxygen, saliva); <b>bottom</b> = deep in the pocket (airless). One wall is the <b>tooth</b>, the opposite wall is the <b>gum</b>. Each dot is a patch of bacteria placed where the engine says it can survive. Turn things on/off on the left.</div>
  <div style="margin:0 0 12px;"><div class="h4">Color key</div><div id="key"></div></div>
  <div class="cake"><div class="h4">The 8-layer cake (highlighted = what you're viewing)</div>
    <div id="ck1">1 · Topology — who lives here &amp; who's linked</div><div id="ck2">2 · Terrain — where they can live (surfaces)</div>
    <div id="ck3">3 · Chemistry — the gradients</div><div id="ck4">4 · Phages — viruses</div><div id="ck5">5 · Defenses</div>
    <div id="ck6">6 · Host immunity — the gum side</div><div id="ck7">7 · OMV cascade — disease to the body</div><div id="ck8">8 · Exogenous — what we add</div></div>
</div>
<div class="hint">tooth wall (cream) · gum wall (pink) · open mouth (blue) · deep pocket (bottom)</div>
<div id="labels"></div>

/*__THREE__*/
<script>
/*__DATA__*/
(function(){
 var THREE=window.THREE; if(!THREE){document.getElementById('exp').innerHTML='3-D engine failed to load (needs internet the first time).';return;}
 var GLAB={reducers:'nitrate-reducers (protective)',red:'red-complex pathogens',pioneers:'pioneers / commensals',bridges:'bridges (connectors)',fungi:'fungi (Candida)',other:'other pathobionts'};
 var EXPL={
  reducers:'<b>Nitrate-reducers</b> (Rothia, Veillonella, Neisseria) — the protective "good" bugs. They turn dietary nitrate into <b>nitric oxide</b>, which relaxes blood vessels. They sit near the surface where there is oxygen.',
  red:'<b>Red-complex pathogens</b> (P. gingivalis, T. denticola, T. forsythia) — the aggressive disease-causers. The engine places them <b>deep and against the gum</b>, where it is airless and iron-rich from bleeding.',
  pioneers:'<b>Pioneers / commensals</b> (streptococci, Actinomyces) — friendly early settlers that stick to the clean tooth first.',
  bridges:'<b>Bridges</b> (Fusobacterium) — the <b>connectors</b> that physically link the harmless surface bugs to the dangerous deep ones. Aim a phage at these to collapse the bad community.',
  fungi:'<b>Fungi</b> (Candida yeast) — fungal members of the community, near the surface.',
  other:'<b>Other pathobionts</b> — aggressive, cariogenic and tumor-associated species (A. actinomycetemcomitans, S. mutans, F. alocis…).',
  connections:'<b>Coaggregation links</b> — who physically sticks to whom (the real network from the catalog). The bad bugs are dangerous because of how they are wired in.',
  interfaces:'<b>The six surfaces</b> — tooth/enamel (cream), gum/soft tissue (pink), the open mouth (blue), and the surrounding biofilm. <b>The surfaces are where all the action happens.</b>',
  omv:'<b>OMVs leaving for the body</b> — the deep bugs near the gum shed tiny toxic bubbles that cross into the tissue and travel to the heart, brain and arteries. The <b>oral → systemic</b> link (Layer 7).',
  grad:'<b>Gradients</b> — the invisible chemistry the engine solved. <b>Oxygen</b> high at the surface, gone deep. <b>pH</b> more neutral/alkaline by the gum. <b>Nitric oxide</b> made by the reducers. <b>Heme</b> (blood iron) pools at the bleeding gum. Slide to move the slice.',
  labels:'<b>Wall labels</b> — the six surfaces named: the tooth (enamel + its protein pellicle), the gum (soft tissue), the open mouth, the deep pocket, and the surrounding biofilm.',
  cma:'<b>C. matruchotii colony</b> — a filament-forming Corynebacterium (the "corncob" backbone). It colonizes the enamel surface by very fast binary fission, then runs a biomineralization program.',
  mineral:'<b>Hydroxyapatite mineral</b> — the calcium-phosphate crystal the colony nucleates from the enamel-supplied Ca/PO₄. This is how dental calculus (tartar) forms.',
  matruchotii:'<b>C. matruchotii — colonize fast, then mineralize.</b> Press play: the colony (gold) spreads across the tooth wall by rapid fission (<b>Phase 1</b>), then once dense it nucleates hydroxyapatite crystal (pale blue) from Ca/PO₄ at the enamel (<b>Phase 2 — calculus</b>). This is the engine stepping forward in time, not a movie.'
 };
 var CMAP={reducers:[1],red:[1],pioneers:[1],bridges:[1],fungi:[1],other:[1],connections:[1],interfaces:[2,6],labels:[2],omv:[7],grad:[3],cma:[2],mineral:[2],matruchotii:[2]};

 var cv=document.getElementById('c');
 var renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});
 renderer.setPixelRatio(Math.min(2,devicePixelRatio||1));
 function size(){renderer.setSize(innerWidth,innerHeight);cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();if(composer)composer.setSize(innerWidth,innerHeight);}
 var scene=new THREE.Scene();
 var cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.1,1000);
 var dist0=26,dist=26,yaw0=0.7,pitch0=0.42,yaw=yaw0,pitch=pitch0;
 function upd(){cam.position.set(dist*Math.cos(pitch)*Math.sin(yaw),dist*Math.sin(pitch),dist*Math.cos(pitch)*Math.cos(yaw));cam.lookAt(0,0,0);}
 var grp=new THREE.Group();scene.add(grp);
 // --- high-end visuals: lighting, atmosphere, glow sprites, bloom ---
 renderer.setClearColor(0x05070b,1);
 scene.add(new THREE.AmbientLight(0x5a6b86,0.95));
 var dir=new THREE.DirectionalLight(0xfff4e6,0.85);dir.position.set(9,15,11);scene.add(dir);
 var pl=new THREE.PointLight(0x86b4ff,0.5,90);pl.position.set(-13,5,9);scene.add(pl);
 scene.fog=new THREE.FogExp2(0x06090e,0.0085);
 (function(){var c=document.createElement('canvas');c.width=2;c.height=256;var x=c.getContext('2d');var g=x.createLinearGradient(0,0,0,256);g.addColorStop(0,'#0e1421');g.addColorStop(1,'#05070b');x.fillStyle=g;x.fillRect(0,0,2,256);scene.background=new THREE.CanvasTexture(c);})();
 var GLOW=(function(){var c=document.createElement('canvas');c.width=c.height=64;var g=c.getContext('2d');var rg=g.createRadialGradient(32,32,0,32,32,32);rg.addColorStop(0,'rgba(255,255,255,1)');rg.addColorStop(0.28,'rgba(255,255,255,0.82)');rg.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=rg;g.fillRect(0,0,64,64);return new THREE.CanvasTexture(c);})();
 function glowMat(color,size){return new THREE.PointsMaterial({color:new THREE.Color(color),size:size,map:GLOW,transparent:true,opacity:0.88,depthWrite:false,blending:THREE.NormalBlending});}
 var composer=null;
 if(THREE.EffectComposer&&THREE.UnrealBloomPass&&THREE.RenderPass){composer=new THREE.EffectComposer(renderer);composer.addPass(new THREE.RenderPass(scene,cam));composer.addPass(new THREE.UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight),0.18,0.3,0.55));}
 var S=10;
 function world(px,py,pz){return [(px-0.5)*S,(pz-0.5)*S,(py-0.5)*S];} // X enamel→gum, Y deep→oral(up), Z biofilm

 grp.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(S,S,S)),new THREE.LineBasicMaterial({color:0x33414f})));
 // faces
 var faces=new THREE.Group();grp.add(faces);
 var faceMeshes={};
 var FACEMAT={enamel:{color:0xf3e6d4,roughness:0.32,metalness:0.06},gum:{color:0xf48ba0,roughness:0.72,metalness:0.0},oral:{color:0x9ad6f5,roughness:0.08,metalness:0.0},deep:{color:0x3d4c63,roughness:0.95,metalness:0.0},biofilm:{color:0x586a80,roughness:0.92,metalness:0.0}};
 function face(key,rx,ry,x,y,z){var p=FACEMAT[key];var m=new THREE.Mesh(new THREE.PlaneGeometry(S,S),new THREE.MeshStandardMaterial({color:p.color,roughness:p.roughness,metalness:p.metalness,transparent:true,opacity:0.1,side:THREE.DoubleSide,depthWrite:false}));m.rotation.x=rx;m.rotation.y=ry;m.position.set(x,y,z);faces.add(m);(faceMeshes[key]=faceMeshes[key]||[]).push(m);}
 face('enamel',0,Math.PI/2,-S/2,0,0);face('gum',0,Math.PI/2,S/2,0,0);face('oral',Math.PI/2,0,0,S/2,0);face('deep',Math.PI/2,0,0,-S/2,0);face('biofilm',0,0,0,0,-S/2);face('biofilm',0,0,0,0,S/2);
 var SURFINFO={
  enamel:{name:'Tooth · enamel + pellicle',col:'#f5d9c0',html:'<b>Native surface:</b> enamel (hydroxyapatite) coated by the <b>acquired pellicle</b> — a thin salivary protein film (statherin, proline-rich proteins, mucins, α-amylase).<br><b>Information here:</b> the pellicle is the <b>adhesion API</b> — which proteins it shows decides which bacteria dock first (S. sanguinis/gordonii bind PRPs via SspA/B; A. oris via FimA). The first-attachment rule lives here.<br><b>Exchange:</b> Ca²⁺ / PO₄³⁻ equilibrium with the enamel — de- and re-mineralization; the substrate for <b>biomineralization</b> (calculus).<br><b>Transports through:</b> none in bulk — solid &amp; anoxic. Only ion exchange (Ca/PO₄) and adhesin–pellicle binding.'},
  gum:{name:'Gum · soft tissue',col:'#fb7185',html:'<b>Native surface:</b> the pocket epithelium &amp; connective tissue — keratinocytes, fibroblasts, with blood + nerve supply and resident immune cells (neutrophils, macrophages).<br><b>Information here:</b> the <b>host–immune interface</b> — senses bacterial LPS/peptides and mounts inflammation; source of gingival crevicular fluid (GCF).<br><b>Exchange:</b> serum-derived fluid flows OUT (pH ≈ 7.4, antibodies, complement, neutrophils, nutrients, and <b>heme</b> when it bleeds); inflammatory mediators (IL-6, TNF-α) cross here; O₂ is low (tissue sink).<br><b>Transports through — the systemic gateway:</b> bacteria translocate INTO blood (<b>bacteremia</b>); <b>OMVs</b> cross into tissue and travel onward; mediators spill into the circulation. The oral→systemic face.'},
  oral:{name:'Open mouth · oral cavity',col:'#7dd3fc',html:'<b>Native surface:</b> the saliva-bathed, air-exposed opening of the pocket to the mouth.<br><b>Information here:</b> the <b>environmental boundary</b> — sets salivary flow, pH and what the diet delivers.<br><b>Exchange:</b> saliva delivers <b>O₂</b> (air-equilibrated), bicarbonate buffering, dietary <b>nitrate</b> (→ nitric oxide), sugars (the Stephan acid pulse), antimicrobial peptides &amp; sIgA — and <b>clears</b> acids/metabolites.<br><b>Transports through:</b> bulk salivary flow — O₂, nitrate, sugar IN; lactate, acids, H₂S and shed planktonic cells OUT.'},
  deep:{name:'Deep pocket',col:'#64748b',html:'<b>Native surface:</b> deeper toward the junctional epithelium — the most anoxic, most protected zone.<br><b>Information here:</b> the <b>refuge</b> — late colonizers &amp; red-complex keystones accumulate; lowest O₂, highest heme.<br><b>Exchange:</b> little flow; diffusion-limited; nutrient/peptide <b>cross-feeding</b> dominates.<br><b>Transports through:</b> in this model a <b>unit-cell / periodic</b> closure (the biofilm continues beyond) — minimal bulk transport.'},
  biofilm:{name:'Adjacent biofilm',col:'#94a3b8',html:'<b>Native surface:</b> the surrounding biofilm continuing sideways — this cube is one <b>representative volume</b> of a larger plaque.<br><b>Information here:</b> continuity with neighboring microcolonies of the same community.<br><b>Exchange:</b> lateral diffusion of metabolites and <b>quorum-sensing</b> signals with the neighbors.<br><b>Transports through:</b> a <b>periodic</b> boundary — what leaves one side re-enters the other (the unit-cell assumption).'}
 };
 var surfOn={},shownSurf=null;
 function setSurface(key,on){surfOn[key]=on;(faceMeshes[key]||[]).forEach(function(m){m.material.opacity=on?0.9:0.12;m.material.depthWrite=!!on;});var b=document.querySelector('.btn[data-s="'+key+'"]');if(b)b.classList.toggle('on',!!on);
  if(on){var inf=SURFINFO[key];document.getElementById('sboxnm').innerHTML='<span class="d" style="background:'+inf.col+'"></span>'+inf.name;document.getElementById('sboxbody').innerHTML=inf.html;document.getElementById('sbox').style.display='block';shownSurf=key;}
  else if(shownSurf===key){document.getElementById('sbox').style.display='none';shownSurf=null;}}
 function overPanel(x,y){var ids=['top','ctl','side','sbox'];for(var i=0;i<ids.length;i++){var el=document.getElementById(ids[i]);if(!el||el.style.display==='none')continue;var r=el.getBoundingClientRect();if(x>=r.left&&x<=r.right&&y>=r.top&&y<=r.bottom)return true;}return false;}
 var LBLS=[{t:'tooth · enamel + pellicle',p:new THREE.Vector3(-S/2-0.4,0,0)},{t:'gum · soft tissue',p:new THREE.Vector3(S/2+0.4,0,0)},{t:'open mouth · saliva, O₂',p:new THREE.Vector3(0,S/2+0.4,0)},{t:'deep pocket',p:new THREE.Vector3(0,-S/2-0.4,0)},{t:'adjacent biofilm',p:new THREE.Vector3(0,0,S/2+0.4)}];
 var labWrap=document.getElementById('labels');
 LBLS.forEach(function(o){var e=document.createElement('div');e.className='lab';e.textContent=o.t;labWrap.appendChild(e);o.el=e;});
 function updLabels(){var on=window._st&&window._st.labels;LBLS.forEach(function(o){if(!on){o.el.style.display='none';return;}var v=o.p.clone().project(cam);var px=(v.x*0.5+0.5)*innerWidth,py=(-v.y*0.5+0.5)*innerHeight;if(v.z>1||overPanel(px,py)){o.el.style.display='none';return;}o.el.style.display='block';o.el.style.left=px+'px';o.el.style.top=py+'px';o.el.style.transform='translate(-50%,-50%)';});}

 // points grouped
 var P=DATA.points, wpos=P.map(function(p){return world(p[0],p[1],p[2]);});
 var objs={};
 DATA.groups.forEach(function(g){
   var pos=[],c=new THREE.Color(DATA.group_color[g]);
   for(var i=0;i<P.length;i++){if(DATA.groups[P[i][3]]===g){pos.push(wpos[i][0],wpos[i][1],wpos[i][2]);}}
   var geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
   var pt=new THREE.Points(geo,glowMat(DATA.group_color[g],g==='fungi'?0.6:0.42));
   grp.add(pt);objs[g]=pt;
 });
 // edges
 var ep=[];DATA.edges.forEach(function(e){var a=wpos[e[0]],b=wpos[e[1]];if(a&&b){ep.push(a[0],a[1],a[2],b[0],b[1],b[2]);}});
 var eg=new THREE.BufferGeometry();eg.setAttribute('position',new THREE.Float32BufferAttribute(ep,3));
 objs.connections=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:0x94a3b8,transparent:true,opacity:0.22}));objs.connections.visible=false;grp.add(objs.connections);
 // OMV particles (drift out through the gum, +X)
 var omvBase=DATA.omv.map(function(i){return wpos[i];}).filter(Boolean), omvPos=[],omvPh=[];
 omvBase.forEach(function(b){omvPos.push(b[0],b[1],b[2]);omvPh.push(Math.random());});
 var og=new THREE.BufferGeometry();og.setAttribute('position',new THREE.Float32BufferAttribute(omvPos,3));
 objs.omv=new THREE.Points(og,glowMat(0xfca5a5,0.42));objs.omv.visible=false;grp.add(objs.omv);
 // time-playback — C. matruchotii colony + the mineral it nucleates (real frames)
 var demo=DATA.demo,FR=demo.frames;
 function ptsObj(arr,color,size){var pos=[];arr.forEach(function(p){var w=world(p[0],p[1],p[2]);pos.push(w[0],w[1],w[2]);});var g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));return new THREE.Points(g,glowMat(color,size));}
 objs.cma=ptsObj([],demo.cma_color,0.45);grp.add(objs.cma);
 objs.mineral=ptsObj([],demo.mineral_color,0.52);grp.add(objs.mineral);
 var curFrame=0,playing=false,ptimer=null;
 function loadFrame(i){curFrame=i;var f=FR[i];
  grp.remove(objs.cma);objs.cma=ptsObj(f.cma,demo.cma_color,0.45);objs.cma.visible=!!state.cma;grp.add(objs.cma);
  grp.remove(objs.mineral);objs.mineral=ptsObj(f.mineral,demo.mineral_color,0.52);objs.mineral.visible=!!state.mineral;grp.add(objs.mineral);
  var biomin=f.min>1;
  document.getElementById('phase').innerHTML=(biomin?'<b style="color:#dbeafe">Phase 2 · biomineralization</b>':'<b style="color:#f0d27a">Phase 1 · colonization</b>')+' &nbsp;t = '+f.t+' h &nbsp;·&nbsp; colony '+f.cma_n.toLocaleString()+' voxels &nbsp;·&nbsp; mineral '+f.min;}
 function setPlay(p){playing=p;document.getElementById('playlbl').textContent=p?'❚❚ pause':'▶ play';document.getElementById('play').classList.toggle('on',p);
  if(p){explain('matruchotii');ptimer=setInterval(function(){var n=curFrame+1;if(n>=FR.length)n=0;document.getElementById('frame').value=n;loadFrame(n);},230);}else{clearInterval(ptimer);}}

 // gradient slice from the real field
 var fn=DATA.field_n;
 var canvas=document.createElement('canvas');canvas.width=canvas.height=fn;var ctx=canvas.getContext('2d');
 var tex=new THREE.CanvasTexture(canvas);tex.minFilter=THREE.LinearFilter;tex.magFilter=THREE.LinearFilter;
 var slice=new THREE.Mesh(new THREE.PlaneGeometry(S,S),new THREE.MeshBasicMaterial({map:tex,transparent:true,opacity:0.82,side:THREE.DoubleSide,depthWrite:false}));slice.rotation.x=-Math.PI/2;slice.visible=false;grp.add(slice);
 function L(a,b,t){return Math.round(a+(b-a)*t);}
 function ramp(n,v){if(n==='O2')return[L(8,70,v),L(22,205,v),L(55,235,v)];if(n==='pH')return v<0.5?[L(232,205,v/0.5),L(70,180,v/0.5),L(60,95,v/0.5)]:[L(205,70,(v-0.5)/0.5),L(180,150,(v-0.5)/0.5),L(95,232,(v-0.5)/0.5)];if(n==='NO')return[L(12,60,v),L(30,232,v),L(25,140,v)];if(n==='heme')return[L(22,205,v),L(8,30,v),L(15,45,v)];return[128,128,128];}
 function drawSlice(){
   var n=document.getElementById('grad').value,d=document.getElementById('depth').value/100;
   if(n==='off'){slice.visible=false;return;}slice.visible=true;slice.position.y=(d-0.5)*S;
   var q=DATA.fields[n].q, k=Math.round(d*(fn-1)), img=ctx.createImageData(fn,fn);
   for(var v=0;v<fn;v++)for(var u=0;u<fn;u++){var idx=(u*fn+v)*fn+k;var val=q[idx]/255;var c=ramp(n,val);var o=(v*fn+u)*4;img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=235;}
   ctx.putImageData(img,0,0);tex.needsUpdate=true;
 }

 // controls
 var state={};DATA.groups.forEach(function(g){state[g]=1;});state.labels=1;state.connections=0;state.omv=0;state.cma=1;state.mineral=1;window._st=state;
 function applyState(){DATA.groups.forEach(function(g){objs[g].visible=!!state[g];});objs.connections.visible=!!state.connections;objs.omv.visible=!!state.omv;objs.cma.visible=!!state.cma;objs.mineral.visible=!!state.mineral;updLabels();}
 function hot(ls){for(var i=1;i<=8;i++){var e=document.getElementById('ck'+i);if(e)e.classList.toggle('hot',ls.indexOf(i)>=0);}}
 function explain(k){var e=document.getElementById('exp');if(EXPL[k])e.innerHTML=EXPL[k];hot(CMAP[k]||[]);}
 // build population buttons + key
 var pb=document.getElementById('popbtns'),ky=document.getElementById('key');
 DATA.groups.forEach(function(g){
   var b=document.createElement('button');b.className='btn on';b.setAttribute('data-k',g);
   b.innerHTML='<span class="d" style="background:'+DATA.group_color[g]+'"></span>'+(GLAB[g]||g);pb.appendChild(b);
   var kk=document.createElement('div');kk.className='key';kk.innerHTML='<span class="d" style="background:'+DATA.group_color[g]+'"></span>'+(GLAB[g]||g);ky.appendChild(kk);
 });
 document.querySelectorAll('.btn[data-k]').forEach(function(b){b.addEventListener('click',function(){var k=b.getAttribute('data-k');state[k]=state[k]?0:1;b.classList.toggle('on',!!state[k]);applyState();explain(k);});});
 document.getElementById('grad').addEventListener('change',function(){drawSlice();explain('grad');});
 document.getElementById('depth').addEventListener('input',drawSlice);
 document.getElementById('resetcam').addEventListener('click',function(){dist=dist0;yaw=yaw0;pitch=pitch0;upd();});
 // time-playback wiring
 var fr=document.getElementById('frame');fr.max=FR.length-1;
 fr.addEventListener('input',function(){if(playing)setPlay(false);loadFrame(+this.value);});
 document.getElementById('play').addEventListener('click',function(){setPlay(!playing);});
 // per-surface toggles (translucent ↔ opaque) + the surface info box
 document.querySelectorAll('.btn[data-s]').forEach(function(b){b.addEventListener('click',function(){var k=b.getAttribute('data-s');setSurface(k,!surfOn[k]);});});
 document.getElementById('sboxx').addEventListener('click',function(){document.getElementById('sbox').style.display='none';shownSurf=null;});
 // reflect default state on the static buttons
 ['labels','connections','omv','cma','mineral'].forEach(function(k){var b=document.querySelector('.btn[data-k="'+k+'"]');if(b)b.classList.toggle('on',!!state[k]);});
 applyState();upd();size();loadFrame(0);

 // orbit
 var drag=false,lx=0,ly=0;
 cv.addEventListener('pointerdown',function(e){drag=true;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId);});
 cv.addEventListener('pointermove',function(e){if(!drag)return;yaw-=(e.clientX-lx)*0.006;pitch=Math.max(-1.35,Math.min(1.35,pitch+(e.clientY-ly)*0.006));lx=e.clientX;ly=e.clientY;upd();});
 cv.addEventListener('pointerup',function(){drag=false;});
 cv.addEventListener('wheel',function(e){e.preventDefault();dist=Math.max(13,Math.min(48,dist+(e.deltaY>0?1.6:-1.6)));upd();},{passive:false});
 addEventListener('resize',size);

 var t=0;
 (function loop(){requestAnimationFrame(loop);
   if(!drag){yaw+=0.0006;upd();}
   if(objs.omv.visible){t+=0.02;var a=og.attributes.position.array;for(var i=0;i<omvBase.length;i++){var ph=(omvPh[i]+t*0.4)%1;a[i*3]=omvBase[i][0]+ph*5.0;a[i*3+1]=omvBase[i][1]+(ph*0.6);a[i*3+2]=omvBase[i][2];}og.attributes.position.needsUpdate=true;}
   if(composer)composer.render();else renderer.render(scene,cam);
   updLabels();
 })();
})();
</script></body></html>"""

