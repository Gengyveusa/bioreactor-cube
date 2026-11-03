// Control-flow and rendering-contract checks, without a browser or external dependency.
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const html=fs.readFileSync('demo/cube-journey-recorded.html','utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
const context2d=new Proxy({createRadialGradient:()=>({addColorStop(){}})}, {get:(o,k)=>o[k]??(()=>{})});
function element(id){return {id,value:({strength:'0.7',start:'20',time:'0'})[id]||'',tagName:'DIV',dataset:{},style:{},textContent:'',disabled:false,files:[],events:{},classList:{toggle(){},add(){},remove(){}},addEventListener(n,fn){this.events[n]=fn},getBoundingClientRect(){return {width:id==='graph'?700:1000,height:id==='graph'?48:800,left:0,top:0}},getContext(){return context2d},setAttribute(){},setPointerCapture(){},append(){},focus(){},scrollIntoView(){},showModal(){this.open=true},close(){this.open=false}}}
const els=Object.fromEntries([...html.matchAll(/id="([\w-]+)"/g)].map(m=>[m[1],element(m[1])]));
const groups={};for(const attr of ['region','move','prompt'])groups['[data-'+attr+']']=[...html.matchAll(new RegExp('data-'+attr+'="([^"]+)"','g'))].map(m=>Object.assign(element(m[1]),{dataset:{[attr]:m[1]}}));
const winEvents={},docEvents={},sandbox={console,Set,Math,JSON,Array,Object,Number,String,Promise,AbortController,devicePixelRatio:1,setTimeout,clearTimeout,requestAnimationFrame(){},ResizeObserver:class{observe(){}},FileReader:class{},fetch:()=>{throw Error('Portable demo must not fetch')},document:{getElementById:id=>{assert.ok(els[id],id);return els[id]},querySelectorAll:s=>groups[s]||[],createElement:()=>element('option'),addEventListener:(n,f)=>docEvents[n]=f,hidden:false},addEventListener:(n,f)=>{const old=winEvents[n];winEvents[n]=old?(e=>{old(e);f(e)}):f}};sandbox.window=sandbox;
vm.createContext(sandbox);for(const script of scripts)vm.runInContext(script,sandbox);
const run=s=>vm.runInContext(s,sandbox);
(async()=>{
assert.equal(run('experiment.parameters.strength'),.7);assert.equal(els.ask.disabled,true);assert.equal(els['source-image'].style.display,'block');
run("setRegion('circulation');setMovement('walk')");assert.equal(run('region'),'circulation');assert.equal(run('camera.p[1]'),14);
run("keys.add('KeyE');motion(1)");assert.equal(run('camera.p[1]'),14);
run("setMovement('fly');keys.add('KeyE');motion(1)");assert.ok(run('camera.p[1]')>14);
run('clearKeys()');winEvents.keydown({target:{tagName:'TEXTAREA'},code:'KeyW'});assert.equal(run('keys.size'),0);
winEvents.keydown({target:{tagName:'CANVAS'},code:'KeyW',preventDefault(){}});assert.equal(run('keys.size'),1);winEvents.blur();assert.equal(run('keys.size'),0);
for(const r of ['oral','barrier','tissue','circulation','plaque'])run(`setRegion('${r}');render(18000)`);
run('overview=true;render(20000)');els.baseline.onclick();assert.equal(run('branch'),'baseline');els.intervention.onclick();assert.equal(run('branch'),'intervention');
els.time.value='100';els.time.oninput();assert.equal(run('step'),100);assert.match(els.delta.textContent,/step 100/);
run('step=120;playing=false');els.play.onclick();assert.equal(run('step'),0);assert.equal(run('playing'),true);
els.strength.value='0';els.start.value='40';await els.run.onclick();assert.equal(run('experiment.parameters.strength'),0);assert.equal(run('experiment.parameters.start'),40);
assert.equal(run('JSON.stringify(experiment.baseline)===JSON.stringify(experiment.intervention)'),true);
els['ref-tongue'].onclick();assert.match(els['source-note'].textContent,/tongue/);
els.tour.onclick();assert.equal(run('touring'),true);run('tick(21000)');els.tour.onclick();assert.equal(run('touring'),false);
// Added v2 user journeys.
run("split=true;setRegion('plaque');render(30000)");assert.equal(run('view.w'),1000);assert.equal(run('branch'),'intervention');
run("split=false;step=120;following=true;followHero()");assert.equal(run('region'),'plaque');assert.ok(run('camera.p[2]')>700);
run("pathway('contamination')");assert.match(els['pathway-context'].textContent,/Endotoxin-free/);
els['ref-plaque'].onclick();assert.match(els['source-note'].textContent,/Truong/);
els['source-image'].onclick();assert.equal(els['source-dialog'].open,true);els['source-close'].onclick();assert.equal(els['source-dialog'].open,false);
els['inspect-omv'].onclick();assert.equal(els.manifest.open,true);
await els.judge.onclick();assert.equal(run('storyIndex'),0);assert.equal(run('experiment.parameters.strength'),.7);
for(let i=0;i<4;i++)els['story-next'].onclick();assert.equal(run('region'),'plaque');assert.equal(run('step'),120);assert.equal(run('split'),true);
await els.challenge.onclick();assert.equal(els.challenge.disabled,false);
await els['learning-predict'].onclick();assert.match(els['learning-run'].textContent,/synthetic in-domain/);assert.equal(run('learningReceipt.run_id'),run('experiment.run_id'));await els['learning-next'].onclick();assert.match(els['learning-error'].textContent,/0.56 pp/);els['learning-ask'].onclick();assert.match(els.question.value,/learning benchmark/);
run("load(window.JOURNEY_SNAPSHOTS['0:40'])");assert.equal(run('learningReceipt'),null);await els['learning-next'].onclick();assert.match(els['learning-run'].textContent,/Recorded assay ranking/);
run("saveNote('test hypothesis',false)");assert.ok(run('notes.length')>0);els['clear-notes'].onclick();assert.equal(run('notes.length'),0);
run('keys.clear();following=true');winEvents.keydown({target:{tagName:'CANVAS'},code:'KeyW',preventDefault(){}});assert.equal(run('following'),false);
// Schooling controls and information-energy-terrain-time state.
els['school-count'].value='64';els['school-information'].value='0.12';els['school-energy'].value='1';els['school-terrain'].value='ridge';els['school-duration'].value='30';els['school-alignment'].value='1.4';els['school-reset'].onclick();els['school-open'].onclick();assert.equal(els['school-dialog'].open,true);run('schoolFrame(1000)');assert.equal(run('school.terrain'),'ridge');assert.match(els['school-stats'].textContent,/64 agents/);els['school-coarse'].onclick();run('drawSchool()');assert.equal(run('schoolCoarse'),true);els['school-disrupt'].onclick();run("scaleInfo('quantum')");assert.match(els['scale-detail'].textContent,/classical CPU/);run('school.elapsed=29.99;schoolPaused=false;schoolFrame(1050)');assert.equal(run('school.elapsed'),30);assert.equal(run('schoolPaused'),true);els['school-close'].onclick();assert.equal(run('schoolOpen'),false);
// Exercise live request dispatch with a stub response; no model-provider connection.
run('metaReady=true'); // recorded mode must still refuse calls.
assert.equal(run('recorded'),true);await els.ask.onclick();
console.log('PASS: portable initialization, 5 region renders, paired worlds, plaque follow, pathway/source lens, source zoom, judge sequence, sensitivity, notebook, overview, walk/fly, input guard, blur, branch selection, time scrubbing, replay, paired preset, source reference, guided tour, offline no-fetch.');
})().catch(e=>{console.error(e);process.exitCode=1});
