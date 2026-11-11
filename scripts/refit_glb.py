"""Re-fit the eyes (and optionally the face length) of a workshop GLB that was ALREADY converted.

    python refit_glb.py face.glb                       -> face_fixed.glb  (auto eye fit + centring)
    python refit_glb.py face.glb out.glb --adjust '{"eyeOffsetMm":[0,0.5,0],"faceLength":1.04}'
    python refit_glb.py input_folder output_folder      -> every .glb in the folder, each fitted on its own

Only the eyeball vertices (bound 100 % to skf_eye_l / skf_eye_r) are moved/scaled; with faceLength != 1 the lower face
is stretched about the eye line. Everything else (UVs, textures, weights, 166-joint rig) is untouched.

Accuracy note: a finished GLB no longer carries the source eyelid bones, so the eye centre is estimated from the
vertices skinned to the eyelid bones. Re-converting from the original pack (batch.py / face_convert.py) uses the real
bones and is more exact; this tool is the quick fix when the original pack is not at hand.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from face_tools.glb import read_glb,accessor,write_glb
from face_tools import eye_fit as ef
from face_convert import parse_adjust
from inspect_glb import verify


def refit(src,dst,adjust=None):
    adjust=adjust or {}
    g,raw=read_glb(Path(src));binary=bytearray(raw)
    assert len(g['meshes'])==1 and len(g['meshes'][0]['primitives'])==1,'Expected a single-primitive workshop GLB.'
    prim=g['meshes'][0]['primitives'][0];at=prim['attributes']
    P=accessor(g,raw,at['POSITION']);J=accessor(g,raw,at['JOINTS_0']).astype(int);W=accessor(g,raw,at['WEIGHTS_0'])
    names=[g['nodes'][i]['name'] for i in g['skins'][0]['joints']]
    idx={n:names.index(n) for n in names if n.startswith(('skf_eye','skf_jaw'))}
    eye_mask={}
    for s in 'lr':
        e=idx['skf_eye_'+s];eye_mask[s]=(J[:,0]==e)&(W[:,0]>.99)
    any_eye=eye_mask['l']|eye_mask['r']
    if not eye_mask['l'].any() or not eye_mask['r'].any():raise ValueError('No eyeball vertices (skf_eye_l/r) found in this GLB.')
    def joint_weight(ids):
        w=np.zeros(len(P))
        for k in range(4):w+=np.where(np.isin(J[:,k],ids),W[:,k],0)
        return w
    lm={'bones':{},'lid':{},'skin':P[~any_eye],'jaw':np.zeros((0,3))}
    for s in 'lr':
        w=joint_weight([idx.get(f'skf_eyelid_{k}_{s}',-1) for k in 'tb'])
        lm['lid'][s]=P[(w>=.5)&~any_eye]
    lm['jaw']=P[(joint_weight([idx['skf_jaw']])>=.5)&~any_eye] if 'skf_jaw' in idx else lm['jaw']
    fit=ef.fit_eyes(lm,adjust,uv_fallback=False)
    if not fit.get('centers'):raise ValueError('Eye fit failed: '+'; '.join(fit.get('warnings',[])))
    radius=np.array(fit['radius'])
    newP=P.copy()
    before={}
    for s in 'rl':
        m=eye_mask[s];c0=(P[m].min(0)+P[m].max(0))/2;r0=(P[m].max(0)-P[m].min(0))/2
        before[s]={'center':c0.tolist(),'radius':r0.tolist()}
        newP[m]=(P[m]-c0)*(radius/np.maximum(r0,1e-9))+np.array(fit['centers'][s])
    fl=ef.face_length_field(lm,fit,adjust.get('faceLength',1.0))
    if fl is not None:
        newP[~any_eye]=fl(newP[~any_eye])
    # write positions back in place (tight packed float32 VEC3, as produced by this toolchain)
    acc=g['accessors'][at['POSITION']];view=g['bufferViews'][acc['bufferView']]
    assert 'byteStride' not in view and acc['componentType']==5126
    start=view.get('byteOffset',0)+acc.get('byteOffset',0)
    binary[start:start+newP.size*4]=newP.astype('<f4').tobytes()
    acc['min']=newP.min(0).tolist();acc['max']=newP.max(0).tolist()
    g.setdefault('extras',{})['eyeRefit']={'method':fit['method'],'before':before,'after':{s:{'center':fit['centers'][s],'radius':fit['radius']} for s in 'rl'},
        'warnings':fit.get('warnings'),'adjust':adjust or None,'metrics':ef.face_metrics(lm,fit)}
    Path(dst).parent.mkdir(parents=True,exist_ok=True)
    write_glb(Path(dst),g,bytes(binary))
    verify(dst)
    return g['extras']['eyeRefit']


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input');p.add_argument('output',nargs='?');p.add_argument('--adjust');a=p.parse_args()
    adj=parse_adjust(a.adjust);src=Path(a.input)
    jobs=[(f,(Path(a.output) if a.output else src)/f.name) for f in sorted(src.glob('*.glb'))] if src.is_dir() else [(src,Path(a.output) if a.output else src.with_name(src.stem+'_fixed.glb'))]
    if not jobs:print('no .glb files found');return 1
    bad=0
    for f,out in jobs:
        try:
            r=refit(f,out,adj);print('OK  %s -> %s  [%s]'%(f.name,out,r['method']))
            for w in r.get('warnings') or []:print('    warning:',w)
        except Exception as e:bad+=1;print('FAIL %s: %s'%(f.name,e))
    return 2 if bad else 0
if __name__=='__main__':sys.exit(main())
