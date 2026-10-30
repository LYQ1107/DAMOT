"""Create an exact local expected tree and a compact, content-free upload plan."""
import subprocess
from common import *

def git(*args):return subprocess.check_output(['git','-C',str(ROOT/'publication')]+list(args))

def main():
    base=git('rev-parse','FETCH_HEAD').decode().strip();tree=git('rev-parse',base+'^{tree}').decode().strip()
    all_rows=git('ls-tree','-rz',base).split(b'\0');known=set();tracked={}
    for row in all_rows:
        if not row:continue
        info,path=row.split(b'\t',1);digest=info.decode().split()[2];known.add(digest)
        if path.startswith(b'experiments/ics_advantage/'):tracked[path.decode()]=digest
    git('read-tree',base);entries=[]
    for path in sorted((ROOT/'delivery/experiments/ics_advantage').rglob('*')):
        if not path.is_file():continue
        relative=str(path.relative_to(ROOT/'delivery'));digest=git('hash-object','-w',str(path)).decode().strip()
        if tracked.get(relative)==digest:continue
        git('update-index','--add','--cacheinfo','100644',digest,relative)
        is_text=path.suffix not in ['.pdf','.png','.npy']
        if is_text:
            try:path.read_bytes().decode('utf-8')
            except UnicodeDecodeError:is_text=False
        entries.append(dict(path=relative,mode='100644',type='blob',sha=digest,local_path=str(path),bytes=path.stat().st_size,
                            text=is_text,known_on_remote=digest in known,inline=is_text and path.stat().st_size<=12000 and digest not in known))
    budget=350000
    for entry in entries:
        if entry['inline']:
            if entry['bytes']>budget:entry['inline']=False
            else:budget-=entry['bytes']
    plan=dict(base=base,base_tree=tree,expected_tree=git('write-tree').decode().strip(),entries=entries)
    dump(ROOT/'publication_plan.json',plan)
    print(json.dumps(dict(base=base,expected_tree=plan['expected_tree'],changed=len(entries),
          known=sum(e['known_on_remote'] for e in entries),inline=sum(e['inline'] for e in entries),
          upload_blobs=sum(not e['inline'] and not e['known_on_remote'] for e in entries))))

if __name__=='__main__':main()
