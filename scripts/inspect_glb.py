"""Check actual GLB geometry, embedded PNGs, atlas bounds and workshop rig."""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from face_tools.glb import read_glb,accessor,world_matrices

def verify(path,max_side=4096):
    path=Path(path);g,raw=read_glb(path)
    assert len(g['buffers'])==1 and not g['buffers'][0].get('uri')
    assert g['buffers'][0]['byteLength']<=len(raw)
    for v in g['bufferViews']:
        assert v.get('buffer',0)==0 and 0<=v.get('byteOffset',0)<=v.get('byteOffset',0)+v['byteLength']<=g['buffers'][0]['byteLength']
    assert len(g['meshes'])==len(g['materials'])==1 and len(g['meshes'][0]['primitives'])==1
    p=g['meshes'][0]['primitives'][0];assert p.get('mode',4)==4 and p['material']==0
    assert not g.get('extensionsRequired'), 'Unsupported required extension must be handled explicitly.'
    a=p['attributes'];assert all(k in a for k in ['POSITION','NORMAL','TEXCOORD_0','JOINTS_0','WEIGHTS_0'])
    v=accessor(g,raw,a['POSITION']);normal=accessor(g,raw,a['NORMAL']);uv=accessor(g,raw,a['TEXCOORD_0']);j=accessor(g,raw,a['JOINTS_0']);weights=accessor(g,raw,a['WEIGHTS_0']);idx=accessor(g,raw,p['indices']).reshape(-1)
    assert len(v)>0 and len(idx)>0 and len(idx)%3==0 and idx.min()>=0 and idx.max()<len(v)
    assert all(len(x)==len(v) for x in [normal,uv,j,weights])
    assert all(np.isfinite(x).all() for x in [v,normal,uv,weights])
    assert uv.min()>=-1e-6 and uv.max()<=1+1e-6
    assert weights.min()>=0 and weights.max()<=1.00001 and np.allclose(weights.sum(1),1,atol=2e-5)
    assert np.max(np.abs(np.linalg.norm(normal,axis=1)-1))<.01
    assert len(g['skins'])==1 and len(g['skins'][0]['joints'])==166
    assert all(0<=i<len(g['nodes']) for i in g['skins'][0]['joints']) and 0<=j.min()<=j.max()<166
    names={g['nodes'][i].get('name') for i in g['skins'][0]['joints']}
    assert {'sk_head','skf_eye_l','skf_eye_r','sk_cervicalspine_02'}<=names
    bind=accessor(g,raw,g['skins'][0]['inverseBindMatrices']);assert bind.shape==(166,16) and np.isfinite(bind).all()
    nodes=[(i,n) for i,n in enumerate(g['nodes']) if 'mesh' in n];assert len(nodes)==1 and nodes[0][1]['mesh']==nodes[0][1]['skin']==0
    worlds=world_matrices(g);matrix=worlds[nodes[0][0]]
    world=(np.c_[v,np.ones(len(v))]@matrix.T)[:,:3];bounds=[world.min(0).tolist(),world.max(0).tolist()]
    size=np.ptp(world,axis=0);assert np.abs(world).max()<10 and .10<size[1]<1 and size[0]<1 and size[2]<1
    images=[]
    for image in g['images']:
        assert 'uri' not in image and image['mimeType']=='image/png'
        view=g['bufferViews'][image['bufferView']];start=view.get('byteOffset',0);payload=raw[start:start+view['byteLength']]
        assert payload.startswith(b'\x89PNG\r\n\x1a\n')
        with Image.open(io.BytesIO(payload)) as im:
            im.load();assert max(im.size)<=max_side and min(im.size)>0;images.append(list(im.size))
    material=g['materials'][0];base=material['pbrMetallicRoughness']['baseColorTexture'];assert base.get('texCoord',0)==0
    assert g['textures'][base['index']]['source']<len(images)
    assert material['alphaMode']=='MASK' and isinstance(material.get('doubleSided'),bool)
    return dict(passed=True,file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),meshes=1,primitives=1,materials=1,vertices=len(v),triangles=len(idx)//3,joints=166,images=images,maxTextureSide=max_side,bounds=bounds,geometryFinite=True,weightsNormalized=True,uvInsideAtlas=True,externalDependencies=False,visualVerified=False,projectJoinVerified=False,gameVerified=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('glb');p.add_argument('--max-side',type=int,default=4096);p.add_argument('--report');a=p.parse_args()
    r=verify(a.glb,a.max_side)
    if a.report:Path(a.report).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(r,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
