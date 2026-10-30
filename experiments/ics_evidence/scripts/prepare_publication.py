"""Prepare an exact Git tree for a frozen delivery; GitHub update uses a lease."""
import json
import subprocess
from pathlib import Path
from evidence_cpu import ROOT


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT/'publication')] + list(args))


def main():
    base=git('rev-parse','FETCH_HEAD').decode().strip()
    base_tree=git('rev-parse',base+'^{tree}').decode().strip()
    tracked={}
    for line in git('ls-tree','-rz',base,'--','experiments/ics_evidence').split(b'\0'):
        if line:
            info,path=line.split(b'\t',1)
            tracked[path.decode()]=info.decode().split()[2]
    git('read-tree',base)
    entries=[]
    for path in sorted((ROOT/'delivery/experiments/ics_evidence').rglob('*')):
        if not path.is_file():continue
        relative=str(path.relative_to(ROOT/'delivery'))
        digest=git('hash-object','-w',str(path)).decode().strip()
        if tracked.get(relative)==digest:continue
        git('update-index','--add','--cacheinfo','100644',digest,relative)
        entry=dict(path=relative,mode='100644',type='blob',sha=digest,local_path=str(path),bytes=path.stat().st_size)
        if path.suffix not in ['.pdf','.png','.npy']:
            # Preserve bytes, including CSV CRLF, in GitHub inline tree content.
            entry['content']=path.read_bytes().decode('utf-8')
        entries.append(entry)
    plan=dict(base=base,base_tree=base_tree,expected_tree=git('write-tree').decode().strip(),entries=entries)
    (ROOT/'publication_plan.json').write_text(json.dumps(plan,ensure_ascii=False))
    print(json.dumps(dict(base=base,base_tree=base_tree,expected_tree=plan['expected_tree'],
                          changed=len(entries),binary=sum('content' not in e for e in entries))))


if __name__=='__main__':main()
