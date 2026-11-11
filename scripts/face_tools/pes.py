from pathlib import Path
import hashlib,json,io,re
import numpy as np
from PIL import Image
from .third_party.pes_file_tools.FmdlFile import FmdlFile
from .glb import read_glb,append_accessor,write_glb,make_eye_texture,add_eyes
from .atlas import merge
from .paths import normalize_texture
from .materials import uses_alpha

def convert(ident,name,zh,*,source_dir,output_dir,template_path,model_factory=FmdlFile,origin=None,max_side=4096,explicit_eye_centers=None,adjust=None):
    source=Path(source_dir)
    origin=origin or {}
    index_path=source/'texture-index.json'
    texture_index=json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else None
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    template=Path(template_path)
    g,raw=read_glb(template);binary=bytearray(raw[:g['buffers'][0]['byteLength']])
    g['nodes'][g['extras']['meshNode']].update(mesh=0,skin=0)
    g['meshes']=[{'name':name,'primitives':[]}];g['materials']=[];g['images']=[];g['textures']=[]
    g['samplers']=[{'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':10497}]
    joint_names=[g['nodes'][i]['name'] for i in g['skins'][0]['joints']]
    material_cache={};missing_bones={};omitted=[];counts=[];cleanup=[];alpha_decisions=[]

    def find_hair_tint():
        env=__import__('os').environ.get('FACE_HAIR_COLOR')
        if env:return tuple(int(x) for x in env.split(','))[:3],'user --hair-color'
        for c in sorted(source.rglob('*_color.png')):
            with Image.open(c) as ci:a=np.array(ci.convert('RGBA'))
            m=a[...,3]>128
            if m.sum()>500:return tuple(int(x) for x in np.median(a[m][:,:3],0)),'median of pack colour map '+re.sub(r'^\d+_','',c.name)
        return (20,15,11),'default dark (no colour map in pack)'

    def embed(path,opaque=False):
        with Image.open(path) as src:im=src.convert('RGBA')
        stem=re.sub(r'^\d+_','',Path(path).stem.lower())
        if not opaque and stem.startswith('hair_parts'):
            arr=np.array(im);msk=arr[...,3]>128
            # hair_parts_bsm_alp stores random strand data in RGB (not a colour); only its alpha is a real card shape.
            if msk.any() and arr[msk][:,:3].std(0).min()>40:
                tint,why=find_hair_tint();arr[...,:3]=tint;im=Image.fromarray(arr)
                alpha_decisions.append({'texture':Path(path).name,'hairNoiseRgbReplacedWithSolid':list(tint),'colourSource':why})
        if opaque:im.putalpha(255)
        stream=io.BytesIO();im.save(stream,format='PNG');png=stream.getvalue();binary.extend(b'\0'*(-len(binary)%4));view=len(g['bufferViews'])
        g['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':len(png)});binary.extend(png)
        g['images'].append({'name':path.stem,'mimeType':'image/png','bufferView':view});g['textures'].append({'source':len(g['images'])-1,'sampler':0})
        return len(g['textures'])-1

    def image_size(path):
        with Image.open(path) as im:return im.size

    selected=[];bone_pos={};eye_centers=explicit_eye_centers;adjust=adjust or {}
    for filename in ['face_high.fmdl','hair_high.fmdl']:
        model=model_factory();model.readFile(source/filename)
        for _b in getattr(model,'bones',[]):
            _g=getattr(_b,'globalPosition',None)
            if _g is not None:bone_pos.setdefault(_b.name,(float(_g.x),float(_g.y),float(_g.z)))
        for mesh in model.meshes:
            material=mesh.materialInstance
            base_name=re.sub(r'\.\d+$','',material.name)
            if base_name not in ['fox_skin_mat','fox_hair_mat','fox_head_shell_mat','fox_hair_transfer_mat','fox_hair_decal_mat']:
                omitted.append({'source':filename,'material':material.name,'triangles':len(mesh.faces)})
                continue
            selected.append((filename,mesh))
    # The socket fit is read from the largest actual skin mesh, not the neck.
    selected.sort(key=lambda item: (item[1].materialInstance.name!='fox_skin_mat',-len(item[1].vertices)))
    # ---- per-face measurements: eye fit, centring and (optional) face length -------------------------------------------
    from . import eye_fit as _ef
    def _skin_pts(mesh):
        P=np.array([[v.position.x,v.position.y,v.position.z] for v in mesh.vertices]);ok=np.isfinite(P).all(1)&(np.abs(P).max(1)<10)
        maps=[{b.name:w for b,w in (v.boneMapping or {}).items()} for v in mesh.vertices]
        return P[ok],[m for m,k in zip(maps,ok) if k]
    landmarks=_ef.collect_landmarks([_skin_pts(me) for fn,me in selected if re.sub(r'\.\d+$','',me.materialInstance.name)=='fox_skin_mat'],bone_pos)
    eye_fit_report=None;face_length_fn=None;face_metrics=None
    if eye_centers is None and adjust.get('eyeFit','auto')!='uv':
        try:
            eye_fit_report=_ef.fit_eyes(landmarks,adjust,uv_fallback=False)
        except Exception as _e:
            eye_fit_report=dict(centers=None,radius=None,method='uv-pole (fallback)',warnings=['eye fit failed: '+str(_e)])
        eye_centers=eye_fit_report.get('centers')
        if eye_centers:
            face_metrics=_ef.face_metrics(landmarks,eye_fit_report)
            face_length_fn=_ef.face_length_field(landmarks,eye_fit_report,adjust.get('faceLength',1.0))
    from .facial_weights import source_has_facial_weights,apply as apply_facial_weights
    _skin_bones={b.name for fn,me in selected if re.sub(r'\.\d+$','',me.materialInstance.name)=='fox_skin_mat' for v in me.vertices for b in (v.boneMapping or {})}
    synth_facial=not source_has_facial_weights(_skin_bones);facial_stats=[];shell_notes=[]
    for filename,mesh in selected:
        m=mesh.materialInstance
        base_name=re.sub(r'\.\d+$','',m.name)
        original_vertices=mesh.vertices
        positions=np.array([[v.position.x,v.position.y,v.position.z] for v in original_vertices])
        lookup={id(v):i for i,v in enumerate(original_vertices)}
        tri=np.array([[lookup[id(v)] for v in face.vertices] for face in mesh.faces])
        # Some original mods collapse disabled cards at Y~185,000 metres.
        # Keep valid head-space geometry; never let those points set the bbox.
        outliers=(~np.isfinite(positions).all(1))|(np.abs(positions).max(1)>10)
        far_faces=outliers[tri].any(1)
        geometric=np.cross(positions[tri[:,1]]-positions[tri[:,0]],positions[tri[:,2]]-positions[tri[:,0]])
        tiny_faces=np.linalg.norm(geometric,axis=1)<1e-12
        keep=~(far_faces|tiny_faces)
        if not keep.all():
            cleanup.append({'source':filename,'material':m.name,'outlyingTriangles':int(far_faces.sum()),
                            'degenerateTriangles':int((tiny_faces&~far_faces).sum()),'retainedTriangles':int(keep.sum())})
        if not keep.any(): continue
        used=np.unique(tri[keep]);remap=np.full(len(original_vertices),-1,dtype=int);remap[used]=np.arange(len(used))
        tri=remap[tri[keep]];positions=positions[used]
        if face_length_fn is not None:positions=face_length_fn(positions)
        vertices=[original_vertices[i] for i in used]
        tex=next((tex for role,tex in m.textures if role=='Base_Tex_SRGB'),None)
        if tex is None:raise ValueError(f'Material {m.name} in {filename} is missing Base_Tex_SRGB.')
        if texture_index is not None and base_name=='fox_hair_transfer_mat':
            _k=normalize_texture(tex.directory.rstrip('/')+'/'+tex.filename).rsplit('.',1)[0]+'.ftex'
            if _k not in texture_index:
                omitted.append({'source':filename,'material':m.name,'triangles':len(mesh.faces),'reason':'hair transfer coefficient texture missing from pack: '+tex.filename});continue
        norm_dir=normalize_texture(tex.directory.rstrip('\\/')+'/')
        match=re.search(r'/face/real/([^/]+)/',norm_dir,re.I)
        assert match, 'Expected a retained player-specific base texture: '+tex.directory
        token=match[1]
        if not ident:ident=int(token) if token.isdigit() else token
        if texture_index is not None:
            key=normalize_texture(tex.directory.rstrip('/')+'/'+tex.filename)
            key=key.rsplit('.',1)[0]+'.ftex'
            image=(source/texture_index[key]).with_suffix('.png')
        else:
            image=source/token/(Path(tex.filename).stem+'.png')
        assert image.is_file(),str(image)
        # Shader semantics and real alpha distinguish opaque skin from Hair2
        # overlays even when both reference the same face texture.
        hair=uses_alpha(m,base_name,image)
        if base_name=='fox_head_shell_mat':
            alpha_decisions.append({'source':filename,'material':m.name,'texture':image.name,
                'shader':getattr(m,'shader',''),'alphaMasked':hair})
        key=(str(image),hair)
        if key not in material_cache:
            image_index=embed(image,opaque=not hair);mi=len(g['materials']);material_cache[key]=mi
            mat={'name':name+'_'+m.name+'_'+token,'alphaMode':'MASK' if hair else 'OPAQUE','doubleSided':hair,
                 'pbrMetallicRoughness':{'baseColorTexture':{'index':image_index},'metallicFactor':0,'roughnessFactor':.85 if hair else .78}}
            if hair:mat['alphaCutoff']=.3333
            g['materials'].append(mat)
        normals=np.array([[v.normal.x,v.normal.y,v.normal.z] for v in vertices]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        uv=np.array([[v.uv[0].u,v.uv[0].v] for v in vertices])
        geo=np.cross(positions[tri[:,1]]-positions[tri[:,0]],positions[tri[:,2]]-positions[tri[:,0]])
        reversed_winding=(geo*normals[tri].mean(1)).sum()<0
        if reversed_winding:tri=tri[:,[0,2,1]]
        joints=np.zeros((len(vertices),4),dtype=np.uint16);weights=np.zeros((len(vertices),4))
        for i,v in enumerate(vertices):
            mappings=sorted((v.boneMapping or {}).items(),key=lambda item:item[1],reverse=True)[:4]
            if not mappings:mappings=[(None,1)]
            for k,(bone,weight) in enumerate(mappings):
                original=bone.name if bone else 'sk_head';target={'sk_chest':'sk_spine_03','sk_neck':'sk_cervicalspine_02'}.get(original,original)
                if target not in joint_names:
                    target='sk_cervicalspine_02' if 'neck' in original or 'clavicle' in original else 'sk_head'
                    missing_bones[original]=target
                joints[i,k]=joint_names.index(target);weights[i,k]=weight
            weights[i]/=weights[i].sum()
        if base_name=='fox_head_shell_mat':
            # The head shell duplicates the skin surface; if it also carries jaw/eyelid weights the importer sees two
            # face-skin candidates and rejects the file. Hair follows the skull, so fold those weights into sk_head.
            _fac=[joint_names.index(n) for n in joint_names if n=='skf_jaw' or n.startswith('skf_eyelid')];_head=joint_names.index('sk_head');_n=0
            for i in range(len(joints)):
                acc={}
                for j,w in zip(joints[i],weights[i]):
                    if w>0:
                        if int(j) in _fac:j=_head;_n+=1
                        acc[int(j)]=acc.get(int(j),0)+w
                for k in range(4):joints[i,k]=0;weights[i,k]=0
                for k,(j,w) in enumerate(sorted(acc.items(),key=lambda kv:-kv[1])[:4]):joints[i,k]=j;weights[i,k]=w
            shell_notes.append({'source':filename,'material':m.name,'facialWeightsFoldedIntoHead':_n})
        if synth_facial and base_name=='fox_skin_mat':
            joints,weights,st=apply_facial_weights(positions,joints,weights,joint_names);facial_stats.append(dict(st,source=filename,material=m.name))
        attrs={kind:append_accessor(g,binary,values,typ,component) for kind,values,typ,component in [
            ('POSITION',positions,'VEC3',5126),('NORMAL',normals,'VEC3',5126),('TEXCOORD_0',uv,'VEC2',5126),
            ('JOINTS_0',joints,'VEC4',5123),('WEIGHTS_0',weights,'VEC4',5126)]}
        g['meshes'][0]['primitives'].append({'attributes':attrs,'indices':append_accessor(g,binary,tri.reshape(-1,1),'SCALAR',5125,34963),'material':material_cache[key],'mode':4})
        counts.append({'source':filename,'material':m.name,'vertices':len(vertices),'triangles':len(tri),'windingReversed':bool(reversed_winding)})
    eye_path=source/'eyes_neutral.png';make_eye_texture(eye_path);eye_tex=embed(eye_path)
    eye_mat=len(g['materials']);g['materials'].append({'name':name+'_Eyes','alphaMode':'OPAQUE','doubleSided':False,
        'pbrMetallicRoughness':{'baseColorTexture':{'index':eye_tex},'metallicFactor':0,'roughnessFactor':.24}})
    eye_fit=add_eyes(g,binary,name,material_index=eye_mat,centers=eye_centers,radius=(eye_fit_report or {}).get('radius'))
    g['buffers']=[{'byteLength':len(binary)}];g['asset']['generator']='Original PES FMDL to Workshop GLB converter'
    g['extras']={'player':name,'playerZh':zh,'sourceArchive':origin.get('archive'),'sourceFaceId':ident,
                 'rig':'Weights mapped to retained V17 skeleton; unsupported auxiliary bones fall back to head/neck.',
                 'coordinateSystem':'Y-up metres, face +Z','eyeScope':'New neutral eyes for standalone display.'}
    intermediate=source/(name+'_分材质源.glb');write_glb(intermediate,g,bytes(binary))
    target=folder/(name+'_工坊单网格.glb');report=merge(intermediate,target,max_side)
    report.update(player=zh,sourceFaceId=ident,sourceFpkSha256=hashlib.sha256((source/'face.fpk').read_bytes()).hexdigest() if (source/'face.fpk').exists() else None,
                  sourceParts=counts,omittedGameShaderOverlays=omitted,auxiliaryBoneFallbacks=missing_bones,eyeFit=eye_fit,syntheticFacialWeights=facial_stats or None,headShellWeightFix=shell_notes or None,
                  originalTextureSizes={str(p.relative_to(source)):image_size(p) for p in source.rglob('*bsm*.png')},
                  auxiliarySourceFiles=[p.name for p in source.glob('*.fmdl') if p.name not in ['face_high.fmdl','hair_high.fmdl']],
                  collapsedGeometryCleanup=cleanup,materialAlphaDecisions=alpha_decisions,gameVerified=False)
    report['eyePlacementMethod']='Explicit reviewed eye centers' if explicit_eye_centers else ((eye_fit_report or {}).get('method') if eye_fit_report and eye_fit_report.get('centers') else 'Original face UV pole and surrounding ring')
    report['eyeFitDetails']=eye_fit_report;report['faceMetrics']=face_metrics;report['faceLengthFactor']=float(adjust.get('faceLength',1.0));report['adjustApplied']=adjust or None
    if origin:report['origin']=origin
    (folder/'模型信息.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

