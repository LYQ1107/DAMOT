"""Use the frozen native evaluator, recording the actual two-epoch endpoint."""
import argparse,json,sys
from pathlib import Path
import evaluate_apr_complement as ev
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from protocol import write_json,write_views

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--domain',required=True);a=ap.parse_args()
    assert json.loads((a.root/'train_complete.json').read_text())['epochs']==2
    original_csv=ev.csv_write;original_dump=ev.dump
    def csv_write(path,rows):
        if Path(path).name=='metrics.csv':
            for row in rows:row['epoch']=2
        return original_csv(path,rows)
    def dump(path,obj):
        if Path(path).name=='complete.json' and isinstance(obj,dict) and 'epoch' in obj:obj['epoch']=2
        return original_dump(path,obj)
    ev.csv_write=csv_write;ev.dump=dump
    out=a.root/'results/stage4'/a.domain;out.mkdir(parents=True,exist_ok=True)
    write_views(ev.Config.fromfile(str(a.root/'config.py')),out/'resolved_views.txt',dict(inference='Native unchanged QuasiDenseTracker; see used_config.py',epochs=2))
    write_json(out/'config_diff.json',dict(training_config=str(a.root/'config.py'),epochs=2,metric_protocol='identical to frozen HF evaluation; no subspace intervention',thresholds=[.05,.30,.20],all_frame_HOTA=True))
    ev.evaluate(a.root,4,a.domain)
if __name__=='__main__':main()
