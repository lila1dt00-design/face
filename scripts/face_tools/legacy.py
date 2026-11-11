from pathlib import Path
from types import SimpleNamespace
import json,zlib,hashlib,io,xml.etree.ElementTree as ET
from PIL import Image
from .third_party.pes_model_blender.ModelFile import readModelFile,ParserSettings
from .pes import convert
from .paths import normalize_texture

def convert_legacy(source,ident,en,zh,origin,*,output_dir,template_path,max_side=4096,explicit_eye_centers=None):
    source=Path(source).resolve()
    original_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file() and p.suffix.lower() in ['.model','.mtl','.dds']}
    texture_index={}
    warnings=[]
    class Adapter:
        def readFile(self,virtual):
            stem='face' if Path(virtual).name.startswith('face') else 'hair'
            actual=source/(stem+'_high_win32.model')
            mtl_path=source/(stem+'.mtl')
            if not actual.is_file():raise ValueError(f'Missing required legacy model file: {actual.name}')
            if not mtl_path.is_file():raise ValueError(f'Missing required legacy material file: {mtl_path.name}')
            model,notes=readModelFile(str(actual),ParserSettings());warnings.extend(notes)
            raw=mtl_path.read_bytes()
            if raw[4:8]==b'ESYS':raw=zlib.decompress(raw[16:])
            materials={m.attrib['name']:m for m in ET.fromstring(raw).findall('material')}
            self.meshes=model.meshes
            for mesh in self.meshes:
                name=mesh.material
                if name not in materials:
                    raise ValueError('Original material is missing: '+name+'; inspect source preview and supply an explicit reviewed repair.')
                selected=materials[name]
                diffuse=next((s.attrib['path'] for s in selected.findall('sampler') if s.attrib.get('name')=='DiffuseMap'),None)
                if name in ['face_phong','head_phong','hair_accessory_hts']:
                    mapped='fox_skin_mat'
                elif name.startswith('hair'):
                    mapped='fox_hair_mat'
                elif name=='eyeocclusion_lambert':mapped='fox_eyeOcclusion_mat'
                else:mapped=name
                texture=[]
                if diffuse and mapped in ['fox_skin_mat','fox_hair_mat']:
                    target_name=Path(diffuse.replace('\\','/')).name
                    file=source/target_name
                    if not file.is_file():
                        file=next((p for p in source.iterdir() if p.is_file() and p.name.lower()==target_name.lower()),file)
                    if not file.is_file():raise ValueError(f'Missing referenced legacy texture: {target_name}')
                    png=file.with_suffix('.png')
                    if not png.exists():
                        raw_image=file.read_bytes()
                        if raw_image[4:8]==b'ESYS':raw_image=zlib.decompress(raw_image[16:])
                        with Image.open(io.BytesIO(raw_image)) as im:im.convert('RGBA').save(png)
                    directory=f'/Asset/model/character/face/real/{ident}/'
                    key=normalize_texture(directory+file.stem+'.ftex')
                    texture_index[key]=png.with_suffix('.ftex').name
                    texture=[('Base_Tex_SRGB',SimpleNamespace(directory=directory,filename=png.name))]
                mesh.materialInstance=SimpleNamespace(name=mapped,textures=texture)
                for vertex in mesh.vertices:
                    if vertex.boneMapping:
                        vertex.boneMapping={mesh.boneGroup.bones[i]:w for i,w in vertex.boneMapping.items()}
            (source/'texture-index.json').write_text(json.dumps(texture_index,indent=2),encoding='utf-8')
    # Discover and decode texture paths before convert() snapshots the index.
    for filename in ['face_high.fmdl','hair_high.fmdl']:Adapter().readFile(source/filename)
    meta=convert(ident,en,zh,source_dir=source,model_factory=Adapter,origin=origin,output_dir=output_dir,template_path=template_path,max_side=max_side,explicit_eye_centers=explicit_eye_centers)
    meta.update(sourceFileSha256=original_hashes,legacyParserWarnings=warnings,sourceFormat='PES .model + original material XML and DDS')
    target=Path(output_dir)/'模型信息.json'
    target.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta

