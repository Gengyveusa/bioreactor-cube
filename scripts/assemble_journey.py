"""Inline the authored v2 extension and evidence catalog into the live HTML."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'web/cube-journey.html'
s=p.read_text()
ext=(ROOT/'web/journey-v2.js').read_text().replace('EVIDENCE_JSON',json.dumps(json.loads((ROOT/'data/journey/evidence.json').read_text()),ensure_ascii=False).replace('<','\\u003c'))
if '/* V2_EXTENSION_START */' in s:
 s=re.sub(r'/\* V2_EXTENSION_START \*/[\s\S]*?/\* V2_EXTENSION_END \*/',lambda m:'/* V2_EXTENSION_START */\n'+ext+'\n/* V2_EXTENSION_END */',s)
else:s=s.replace('/* V2_EXTENSION */','/* V2_EXTENSION_START */\n'+ext+'\n/* V2_EXTENSION_END */')
learning=(ROOT/'web/journey-learning.js').read_text().replace('LEARNING_BENCHMARK_JSON',json.dumps(json.loads((ROOT/'data/learning/benchmark.json').read_text())).replace('<','\\u003c'))
s=re.sub(r'/\* LEARNING_EXTENSION_START \*/[\s\S]*?/\* LEARNING_EXTENSION_END \*/',lambda m:'/* LEARNING_EXTENSION_START */\n'+learning+'\n/* LEARNING_EXTENSION_END */',s)
school=(ROOT/'web/schooling.js').read_text()+'\n'+(ROOT/'web/journey-scales.js').read_text().replace('SCALES_JSON',json.dumps(json.loads((ROOT/'data/learning/scales.json').read_text())).replace('<','\\u003c'))
s=re.sub(r'/\* SCHOOL_EXTENSION_START \*/[\s\S]*?/\* SCHOOL_EXTENSION_END \*/',lambda m:'/* SCHOOL_EXTENSION_START */\n'+school+'\n/* SCHOOL_EXTENSION_END */',s)
p.write_text(s)
