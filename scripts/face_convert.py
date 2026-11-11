"""Portable PES head-to-workshop GLB conversion. All preparation happens in a fresh work directory."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
from face_tools.containers import cpk_ids,extract_cpk,unpack_archive,unpack_fpk,resolve_textures
from face_tools.glb import read_glb,world_matrices
from face_tools.pes import convert
from face_tools.legacy import convert_legacy
from face_tools.atlas import merge
from inspect_glb import verify

SKILL=Path(__file__).resolve().parent.parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(1024*1024):h.update(data)
    return h.hexdigest()

def snapshot(path):
    path=Path(path)
    files=path.rglob('*') if path.is_dir() else [path]
    return {str(p.resolve()):sha(p) for p in files if p.is_file()}

def choose_folder(root,face_id=None,subdir=None):
    root=Path(root)
    if subdir:
        original_root=root.resolve();root=(root/subdir).resolve();root.relative_to(original_root)
    candidates=list(root.rglob('face_high.fmdl'))+list(root.rglob('face_high_win32.model'))
    if not candidates:raise ValueError('No supported face_high.fmdl or face_high_win32.model found. Inspect archive format before choosing another adapter.')
    if face_id is not None:
        candidates=[p for p in candidates if str(face_id) in p.parts]
    if len(candidates)!=1:raise ValueError('Multiple or zero matching face folders; use the verified --face-id or --subdir instead of choosing the first result.')
    return candidates[0].parent

def infer_face_id(folder,root):
    rel='/'.join(Path(folder).resolve().relative_to(Path(root).resolve()).parts)
    if m:=re.search(r'(?:^|/)face/real/([^/]+)(?:/|$)',rel,re.I):return m[1]
    if Path(folder).name.isdigit():return Path(folder).name
    return None

def parse_eye_centers(raw):
    if not raw:return None
    text=str(raw).strip()
    if not text.startswith('{'):
        path=Path(text)
        if not path.is_file():raise FileNotFoundError(f'Eye centers JSON file not found: {text}')
        text=path.read_text(encoding='utf-8-sig')
    data=json.loads(text)
    if not isinstance(data,dict) or any(side not in data or len(data[side])!=3 for side in ('r','l')):
        raise ValueError('Eye centers JSON must provide 3D coordinates for "r" and "l".')
    return {'r':[float(x) for x in data['r']],'l':[float(x) for x in data['l']]}

def parse_adjust(raw):
    """Per-face tuning: JSON string or JSON file. Keys: eyeFit('auto'|'uv'), eyeCentering(bool), eyeScale, eyeSeparationScale,
    eyeOffsetMm[x,y,z], faceLength (0.90-1.12)."""
    if not raw:return {}
    if isinstance(raw,dict):text=json.dumps(raw)
    else:text=str(raw).strip()
    if not text.startswith('{'):
        path=Path(text)
        if not path.is_file():raise FileNotFoundError(f'Adjust JSON file not found: {text}')
        text=path.read_text(encoding='utf-8-sig')
    data=json.loads(text)
    if not isinstance(data,dict):raise ValueError('Adjust JSON must be an object.')
    allowed={'eyeFit','eyeCentering','eyeScale','eyeSeparationScale','eyeOffsetMm','faceLength'}
    bad=set(data)-allowed
    if bad:raise ValueError('Unknown adjust keys: '+', '.join(sorted(bad))+'; allowed: '+', '.join(sorted(allowed)))
    if not .5<=float(data.get('eyeScale',1))<=1.6:raise ValueError('eyeScale must be 0.5-1.6')
    if not .5<=float(data.get('eyeSeparationScale',1))<=1.5:raise ValueError('eyeSeparationScale must be 0.5-1.5')
    if not .90<=float(data.get('faceLength',1))<=1.12:raise ValueError('faceLength must be 0.90-1.12')
    if len(data.get('eyeOffsetMm',[0,0,0]))!=3 or max(abs(float(x)) for x in data.get('eyeOffsetMm',[0,0,0]))>10:raise ValueError('eyeOffsetMm must be [x,y,z] within +-10 mm')
    return data

def run(a):
    import os
    if getattr(a,'hair_color',None):os.environ['FACE_HAIR_COLOR']=a.hair_color
    source=Path(a.input).resolve();output=Path(a.output_dir).resolve();work=Path(a.work_dir).resolve()
    if not source.exists():raise FileNotFoundError(source)
    if output.exists() or work.exists():raise FileExistsError('Output and work directories must be new; choose a revision name to preserve existing files.')
    if not re.fullmatch('[A-Za-z0-9_-]+',a.name):raise ValueError('--name must be a stable ASCII file stem, e.g. Pogba2015.')
    if not 64<=a.max_texture_side<=4096:raise ValueError('Workshop atlas bound must be between 64 and 4096.')
    # Never prepare output inside a source folder; snapshots must continue to describe original data only.
    if source.is_dir():
        for p in [work,output]:
            if p==source or source in p.parents:raise ValueError('Work/output must be outside the original source directory.')
    if output in work.parents or work==output or work in output.parents:raise ValueError('Keep output and work directories separate.')
    before=snapshot(source);work.mkdir(parents=True);prepared=work/'prepared_input'
    provenance=json.loads(Path(a.provenance).read_text(encoding='utf-8-sig')) if a.provenance else {}
    provenance.update(player=a.player,requestedAppearance=a.appearance,sourceInput=str(source),sourceFilesSha256=before)
    chosen_face_id=a.face_id
    if a.source_page:provenance['page']=a.source_page
    target=output/(a.name+'_工坊单网格.glb')
    if source.suffix.lower()=='.glb':
        prepared.mkdir();copy=prepared/source.name;shutil.copy2(source,copy);g,raw=read_glb(copy)
        assert len(g.get('skins',[]))==1 and len(g['skins'][0]['joints'])==166, 'A generic GLB needs explicit rig adaptation; this branch only repacks an existing workshop rig.'
        nodes=[i for i,n in enumerate(g['nodes']) if 'mesh' in n]
        assert len(g['meshes'])==len(nodes)==1 and np.allclose(world_matrices(g)[nodes[0]],np.eye(4)), 'Flatten node transforms before atlas merging.'
        output.mkdir(parents=True);info=merge(copy,target,a.max_texture_side)
    else:
        if source.is_dir():shutil.copytree(source,prepared)
        elif source.suffix.lower()=='.cpk':
            ids=cpk_ids(source);ident=a.face_id or (ids[0] if len(ids)==1 else None)
            if not ident:raise ValueError('CPK contains multiple face IDs; inspect it and choose the player ID explicitly: '+str(ids[:40]))
            chosen_face_id=ident
            extract_cpk(source,ident,prepared)
        elif source.suffix.lower()=='.fpk':
            prepared.mkdir()
            # FPK commonly stores geometry only. Its sibling texture tree must accompany it.
            shutil.copytree(source.parent,prepared/'source_tree');source_copy=prepared/'source_tree'/source.name
            unpack_fpk(source_copy,source_copy.parent)
        elif source.suffix.lower()=='.fmdl':
            prepared.mkdir();shutil.copytree(source.parent,prepared/'source_tree')
        else:unpack_archive(source,prepared,a.member_prefix)
        for fpk in list(prepared.rglob('face.fpk')):unpack_fpk(fpk,fpk.parent)
        # Some mod archives contain one or more CPKs rather than loose geometry.
        candidates=list(prepared.rglob('face_high.fmdl'))+list(prepared.rglob('face_high_win32.model'))
        if not candidates:
            cpks=list(prepared.rglob('*.cpk'))
            if a.cpk_member:
                selected=(prepared/a.cpk_member).resolve();selected.relative_to(prepared.resolve());cpks=[selected] if selected.is_file() else []
            if len(cpks)!=1:raise ValueError('Select a supported face folder or a specific CPK from the archive.')
            ids=cpk_ids(cpks[0]);ident=a.face_id or (ids[0] if len(ids)==1 else None)
            if not ident:raise ValueError('Multiple player IDs in nested CPK: '+str(ids[:40]))
            chosen_face_id=ident
            extract_cpk(cpks[0],ident,prepared/'cpk_contents')
        folder=choose_folder(prepared,a.face_id,a.subdir)
        if not chosen_face_id:chosen_face_id=infer_face_id(folder,prepared)
        eyes=parse_eye_centers(a.eye_centers)
        template=Path(a.template).resolve() if a.template else SKILL/'assets/workshop-rig.glb'
        kwargs=dict(output_dir=output,template_path=template,max_side=a.max_texture_side,explicit_eye_centers=eyes)
        adjust=parse_adjust(getattr(a,'adjust',None))
        ident=int(chosen_face_id) if chosen_face_id and str(chosen_face_id).isdigit() else (chosen_face_id or 0)
        if (folder/'face_high.fmdl').exists():
            resolve_textures(folder,prepared)
            if not (folder/'hair_high.fmdl').exists():raise ValueError('Missing companion hair_high.fmdl; do not borrow another player hair mesh silently.')
            info=convert(ident,a.name,a.player,source_dir=folder,origin=provenance,adjust=adjust,**kwargs)
        else:info=convert_legacy(folder,ident,a.name,a.player,provenance,**kwargs)
    checks=verify(target,a.max_texture_side)
    assert snapshot(source)==before, 'Original input changed during conversion.'
    if not chosen_face_id and not info.get('sourceFaceId'):info['sourceFaceId']=None
    info.update(origin=provenance,verification=checks,sourceUnmodified=True,visualVerified=False,actualWorkshopParserPassed=False,projectJoinVerified=False,gameVerified=False)
    (output/'模型信息.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    (work/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    return dict(file=str(target),passed=True,vertices=checks['vertices'],triangles=checks['triangles'],joints=166,images=checks['images'],sourceUnmodified=True,visualVerified=False,projectJoinVerified=False,gameVerified=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('inspect-cpk');q.add_argument('input')
    c=sub.add_parser('convert');c.add_argument('--input',required=True);c.add_argument('--player',required=True);c.add_argument('--name',required=True)
    c.add_argument('--output-dir',required=True);c.add_argument('--work-dir',required=True);c.add_argument('--face-id');c.add_argument('--subdir');c.add_argument('--member-prefix');c.add_argument('--cpk-member')
    c.add_argument('--source-page');c.add_argument('--appearance',default='');c.add_argument('--provenance');c.add_argument('--eye-centers');c.add_argument('--adjust',help='per-face tuning JSON string/file: eyeFit,eyeCentering,eyeScale,eyeSeparationScale,eyeOffsetMm,faceLength');c.add_argument('--template');c.add_argument('--max-texture-side',type=int,default=4096);c.add_argument('--hair-color',help='R,G,B override for hair, e.g. 20,15,11')
    a=p.parse_args();result=cpk_ids(a.input) if a.action=='inspect-cpk' else run(a);print(json.dumps(result,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
