"""One-click batch mode: converts everything in ./input into ./output (one folder per face)."""
import argparse,hashlib,os,re,shutil,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=Path(os.environ['FACE_ROOT']) if os.environ.get('FACE_ROOT') else HERE.parent
sys.path.insert(0,str(HERE))
from face_tools.containers import cpk_ids,unpack_archive
from face_convert import run

ARCHIVES={'.zip','.rar','.7z'};SINGLE={'.cpk','.fpk','.fmdl','.glb'};MARKERS=('face_high.fmdl','face_high_win32.model')

def safe_name(text,fallback='face'):
    s=re.sub('_+','_',re.sub('[^A-Za-z0-9_-]','_',text)).strip('_')[:60]
    return s or fallback+'_'+hashlib.sha1(text.encode('utf8')).hexdigest()[:6]

def face_folders(root):
    found={p.parent for m in MARKERS for p in root.rglob(m)}|{p.parent for p in root.rglob('face.fpk')}
    # A folder holding face.fpk next to an already-unpacked copy of it (e.g. face_fpk/) is one face, not two.
    found={f for f in found if not ((f/'face.fpk').exists() and any(o!=f and f in o.parents for o in found))}
    return sorted(found)

def plan_dir(root,base):
    """Return a list of (name, kwargs) jobs for an unpacked folder."""
    folders=face_folders(root)
    if folders:
        if len(folders)==1:return [(base,dict(subdir=None if folders[0]==root else str(folders[0].relative_to(root))))]
        return [(base+'_'+safe_name(str(f.relative_to(root)) if f!=root else 'root'),dict(subdir=None if f==root else str(f.relative_to(root)))) for f in folders]
    jobs=[];cpks=sorted(root.rglob('*.cpk'))
    for c in cpks:
        rel=str(c.relative_to(root));ids=cpk_ids(c)
        for i in ids:
            label=i if len(cpks)==1 else safe_name(c.stem)+'_'+i
            jobs.append((base if len(cpks)==1 and len(ids)==1 else base+'_'+label,dict(cpk_member=rel,face_id=i)))
    return jobs

def plan(item,scratch):
    """Return (input_path, jobs). input_path may be a scratch copy of an archive's contents."""
    base=safe_name(item.stem if item.is_file() else item.name);ext=item.suffix.lower()
    if item.is_dir():return item,plan_dir(item,base)
    if ext=='.glb':return item,[(base,{})]
    if ext=='.cpk':
        ids=cpk_ids(item)
        if not ids:raise ValueError('No /face/real/<id>/ entries found in this CPK.')
        return item,[(base if len(ids)==1 else base+'_'+i,dict(face_id=i)) for i in ids]
    if ext in ('.fpk','.fmdl'):return item,[(base,{})]
    if ext in ARCHIVES:
        out=scratch/'unpacked';unpack_archive(item,out)
        for _ in range(2):  # one level of archive-inside-archive
            if face_folders(out) or list(out.rglob('*.cpk')):break
            inner=[p for p in out.rglob('*') if p.is_file() and p.suffix.lower() in ARCHIVES]
            for k,p in enumerate(inner):unpack_archive(p,out/('_inner%d'%k))
        return out,plan_dir(out,base)
    raise ValueError('Unsupported type: '+ext)

def load_adjust(inp):
    """input/adjust.json = {"_default": {...}, "<face name>": {...}}  and/or input/<face name>.adjust.json"""
    import json
    p=inp/'adjust.json'
    try:return json.loads(p.read_text(encoding='utf-8-sig')) if p.is_file() else {}
    except Exception as e:raise SystemExit('input/adjust.json is not valid JSON: %s'%e)

def adjust_for(table,inp,name,item_stem):
    import json
    merged=dict(table.get('_default',{}))
    for key in (item_stem,name):
        if isinstance(table.get(key),dict):merged.update(table[key])
        f=inp/(key+'.adjust.json')
        if f.is_file():merged.update(json.loads(f.read_text(encoding='utf-8-sig')))
    return merged

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-texture-side',type=int,default=4096);ap.add_argument('--max-pixels',type=int,default=4194304,help='cap on total atlas pixels (default 4194304 = 2048x2048)')
    a=ap.parse_args();os.environ['FACE_MAX_PIXELS']=str(a.max_pixels);inp=ROOT/'input';out=ROOT/'output';tmp=ROOT/'_work'
    inp.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    items=sorted(p for p in inp.iterdir() if not p.name.startswith(('.','_')) and (p.is_dir() or p.suffix.lower() in ARCHIVES|SINGLE))
    if not items:
        print('input folder is empty. Put .zip/.rar/.7z/.cpk/.fpk/.fmdl/.glb files (or face folders) in:\n  '+str(inp));return 1
    log=[];ok=fail=skip=0;adj_table=load_adjust(inp)
    for item in items:
        print('\n=== '+item.name);scratch=tmp/safe_name(item.name)
        shutil.rmtree(scratch,ignore_errors=True);scratch.mkdir(parents=True)
        try:src,jobs=plan(item,scratch)
        except Exception as e:
            fail+=1;log.append('FAIL  %s : %s'%(item.name,e));print('  FAIL: %s'%e);traceback.print_exc(file=open(out/'_errors.log','a',encoding='utf8'));shutil.rmtree(scratch,ignore_errors=True);continue
        if not jobs:
            fail+=1;log.append('FAIL  %s : no supported face model found inside'%item.name);print('  FAIL: no supported face model found inside');shutil.rmtree(scratch,ignore_errors=True);continue
        for name,kw in jobs:
            dest=out/name
            if dest.exists():skip+=1;log.append('SKIP  %s : output/%s already exists (delete it to redo)'%(name,name));print('  skip %s (already converted)'%name);continue
            work=scratch/('w_'+name)
            ns=SimpleNamespace(input=str(src),player=name,name=name,output_dir=str(dest),work_dir=str(work),face_id=kw.get('face_id'),subdir=kw.get('subdir'),member_prefix=None,cpk_member=kw.get('cpk_member'),source_page=None,appearance='',provenance=None,eye_centers=None,adjust=(adjust_for(adj_table,inp,name,safe_name(item.stem if item.is_file() else item.name)) or None),template=None,max_texture_side=a.max_texture_side)
            t=time.time()
            try:
                try:
                    os.environ.pop('FACE_SKIP_MISSING_TRANSFER',None);r=run(ns);note=''
                except ValueError as e1:
                    if 'hair_' not in str(e1) or 'coeff' not in str(e1):raise
                    shutil.rmtree(dest,ignore_errors=True);shutil.rmtree(work,ignore_errors=True)
                    os.environ['FACE_SKIP_MISSING_TRANSFER']='1';r=run(ns);note=' NOTE: pack lacks hair-transfer coeff map; its colour strand map was used instead (see 模型信息.json)'
                ok+=1;log.append('OK    %s -> output/%s/  (%d vertices, %d triangles)%s'%(name,name,r['vertices'],r['triangles'],note));print('  OK   %s  (%.0fs)%s'%(name,time.time()-t,note))
            except Exception as e:
                fail+=1;shutil.rmtree(dest,ignore_errors=True);log.append('FAIL  %s : %s: %s'%(name,type(e).__name__,e));print('  FAIL %s: %s'%(name,e))
                with open(out/'_errors.log','a',encoding='utf8') as f:f.write('--- %s\n%s\n'%(name,traceback.format_exc()))
        shutil.rmtree(scratch,ignore_errors=True)
    shutil.rmtree(tmp,ignore_errors=True)
    summary='Done: %d converted, %d failed, %d skipped'%(ok,fail,skip)
    (out/'_report.txt').write_text(summary+'\n\n'+'\n'.join(log)+'\n',encoding='utf8')
    print('\n'+summary+'\nDetails: output/_report.txt'+('  |  errors: output/_errors.log' if fail else ''));return 0 if not fail else 2

if __name__=='__main__':sys.exit(main())
