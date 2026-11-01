"""Launch the unchanged HF stage trainer with two capability-specific check adapters."""
import json
import os
from pathlib import Path
import sys
import runtime_adapter as adapter
Q=Path(__file__).resolve().parents[1]
assert json.loads((Q/'preflight/validation.json').read_text())['status']=='passed'
assert json.loads((Q/'preflight/V3_code_snapshot.json').read_text())['status']=='passed'
record=adapter.install_train_compatibility()
rank=int(os.environ.get('RANK',0))
adapter.hf.dump(Q/'preflight'/('runner_compatibility_rank%d.json'%rank),record)
sys.argv=[str(Q/'code/tools/run/apr_complement_experiment.py'),'--run-root',str(Q),'--mode','train']
adapter.hf.main()
