#!/usr/bin/env python3
"""Freeze one common dense real-frame stream for all module conditions."""
import json, shutil
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
OUT=RUN/'results/exp4'
p=OUT/'active_profile_frame_manifest.json'
x=json.loads(p.read_text())
shutil.copyfile(p, OUT/'active_profile_frame_manifest_postwarm32.json')
top=x['frames'][:4]
assert len(top)==4 and len({r['image_id'] for r in top})==4
x['frames']=top*8
x['selection']='Common 32-entry stream for history 0/1/2/3: the four highest post-warm-up native-positive-RoI real Overcast frames repeated eight times. Distinct source frames are fixed by the 512-frame post-warm-up scan; no GT selection. Repetition keeps batch4 at the unchanged ICS min_samples=96 gate after the 20-step real warm-up. Separate active-branch microbenchmark, not Night mean throughput.'
x['stream_distinct_frames']=4
x['stream_entries']=32
x['source_postwarm_manifest']='active_profile_frame_manifest_postwarm32.json'
x['previous_failures']=x.get('previous_failures',[])+['history3 with postwarm-ranked 32-entry stream: no timed-window covariance update']
p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
print(json.dumps({'stream_distinct_frames':4,'stream_entries':32,'image_ids':[r['image_id'] for r in top]},ensure_ascii=False))
