"""Synthesise jaw / eyelid skin weights for faces whose source rig has none.

Some packs ship a face skinned only to head/neck. The workshop importer identifies the face skin by its soft
weights on skf_jaw / skf_eyelid_*, so such a face is rejected. All PES faces share one head-space coordinate frame, so
we copy the spatial weight *field* (averaged on a 4 mm voxel grid from faces that do carry the rig) onto the vertices.
Only used when the source has no facial-bone weights at all; faces that already have them are never touched."""
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

CHANNELS=['skf_jaw','skf_eyelid_t_l','skf_eyelid_t_r','skf_eyelid_b_l','skf_eyelid_b_r']
FIELD=Path(__file__).resolve().parent.parent.parent/'assets'/'face-weight-field.npz'
MAX_DIST=0.008

def source_has_facial_weights(bone_names):
    return any(n=='skf_jaw' or n.startswith('skf_eyelid') for n in bone_names)

def apply(positions,joints,weights,joint_names,field_path=FIELD):
    """Return (joints, weights, stats). Arrays are (n,4); weights stay normalised."""
    data=np.load(field_path);tree=cKDTree(data['centers']);dist,idx=tree.query(positions)
    f=data['weights'][idx]*(dist<=MAX_DIST)[:,None]                      # (n,5)
    f=np.where(f<0.02,0,f);tot=f.sum(1);over=tot>1;f[over]/=tot[over,None]
    chan=[joint_names.index(c) for c in CHANNELS]
    out_j=np.zeros_like(joints);out_w=np.zeros_like(weights)
    for i in range(len(positions)):
        acc={}
        keep=1-f[i].sum()
        for j,w in zip(joints[i],weights[i]):
            if w>0:acc[int(j)]=acc.get(int(j),0)+w*keep
        for c,w in zip(chan,f[i]):
            if w>0:acc[c]=acc.get(c,0)+w
        top=sorted(acc.items(),key=lambda kv:-kv[1])[:4];s=sum(w for _,w in top)
        for k,(j,w) in enumerate(top):out_j[i,k]=j;out_w[i,k]=w/s
    return out_j,out_w,{'verticesWithFacialWeight':int((f.sum(1)>0).sum()),'meanJawWeight':float(f[:,0].mean())}
