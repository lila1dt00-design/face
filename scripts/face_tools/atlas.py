"""Bake an embedded texture atlas and one primitive for Workshop head preview."""
from pathlib import Path
import io
import json
import hashlib
import numpy as np
from PIL import Image
from .glb import read_glb, accessor, append_accessor, write_glb
from .textures import limit_image


def merge(source_path, target_path, max_side=4096):
    g, raw = read_glb(source_path)
    primitives = [p for mesh in g['meshes'] for p in mesh['primitives']]
    # Pack source images at their native pixel sizes, with an 8-pixel gutter.
    images = []
    for im in g['images']:
        view = g['bufferViews'][im['bufferView']]; start = view.get('byteOffset', 0)
        with Image.open(io.BytesIO(raw[start:start+view['byteLength']])) as src:
            images.append(src.convert('RGBA'))
    # Bake REPEAT sampling into the atlas when source cards extend past 0..1.
    # Modulo-per-vertex would interpolate incorrectly across triangle seams.
    bounds={i:[np.array([0.,0.]),np.array([1.,1.])] for i in range(len(images))}
    for p in primitives:
        material=g['materials'][p['material']]
        tex_index=material['pbrMetallicRoughness']['baseColorTexture']['index']
        image_index=g['textures'][tex_index]['source']
        uv=accessor(g,raw,p['attributes']['TEXCOORD_0'])
        bounds[image_index][0]=np.minimum(bounds[image_index][0],uv.min(0))
        bounds[image_index][1]=np.maximum(bounds[image_index][1],uv.max(0))
    transforms={};repeat_bakes=[]
    for i,im in enumerate(images):
        minimum,maximum=bounds[i]
        low=np.floor(minimum+1e-7).astype(int);high=np.ceil(maximum-1e-7).astype(int)
        tiles=np.maximum(high-low,1)
        assert np.max(tiles)<=8, 'Unexpectedly large source UV tiling'
        transforms[i]=(low,tiles)
        if np.any(tiles!=1) or np.any(low!=0):
            images[i]=Image.fromarray(np.tile(np.array(im),(int(tiles[1]),int(tiles[0]),1)))
            repeat_bakes.append({'image':i,'tileOffset':low.tolist(),'tileCount':tiles.tolist()})
    max_source_side = max(max(im.size) for im in images)
    width = 1
    while width < max(max_source_side+16, sum(im.width+16 for im in images)**.5*max_source_side**.5): width *= 2
    width = max(width, 512)
    boxes = []; x = y = row_height = 0
    for im in images:
        if x+im.width+16 > width: x = 0; y += row_height; row_height = 0
        boxes.append((x+8, y+8, im.width, im.height)); x += im.width+16; row_height = max(row_height, im.height+16)
    height = 1
    while height < y+row_height: height *= 2
    atlas = Image.new('RGBA', (width, height), (0,0,0,0))
    for im, (x,y,w,h) in zip(images, boxes):
        # Repeat each border texel into the gutter without resampling the image.
        padded=Image.fromarray(np.pad(np.array(im),((8,8),(8,8),(0,0)),mode='edge'))
        atlas.paste(padded,(x-8,y-8))
    # Native APK textures accept at most 4096 pixels per side. Keep the
    # normalized atlas UVs below unchanged when downsampling the whole atlas.
    source_atlas_size = atlas.size
    atlas = limit_image(atlas, max_side)
    rough = Image.new('RGB', source_atlas_size, (255,200,0))
    pos=[];norm=[];uvs=[];joints=[];weights=[];colors=[];indices=[];base=0
    original_vertices=original_triangles=0;sided_parts=[]
    for primitive_index,p in enumerate(primitives):
        a = p['attributes']; material = g['materials'][p['material']]
        pbr = material['pbrMetallicRoughness']; tex_index = pbr['baseColorTexture']['index']
        img_index = g['textures'][tex_index]['source'];x,y,w,h = boxes[img_index]
        uv = accessor(g, raw, a['TEXCOORD_0']).astype(float)
        low,tiles=transforms[img_index]
        uv=(uv-low)/tiles
        assert uv.min() >= -1e-6 and uv.max() <= 1+1e-6
        uv[:,0] = (x+uv[:,0]*w)/width;uv[:,1] = (y+uv[:,1]*h)/height
        n = len(uv)
        position=accessor(g,raw,a['POSITION']);normal=accessor(g,raw,a['NORMAL'])
        joint=accessor(g,raw,a['JOINTS_0']);weight=accessor(g,raw,a['WEIGHTS_0'])
        color=accessor(g,raw,a['COLOR_0']) if 'COLOR_0' in a else np.ones((n,4))
        triangles=accessor(g,raw,p['indices']).reshape(-1,3)
        original_vertices+=n;original_triangles+=len(triangles)
        if material.get('doubleSided',False):
            # A single two-sided atlas material would also expose the reverse
            # of skin/neck surfaces. Encode only the intended hair backs.
            sided_parts.append({'primitive':primitive_index,'material':material.get('name'),
                'addedVertices':n,'addedTriangles':len(triangles)})
            position=np.concatenate([position,position]);normal=np.concatenate([normal,-normal])
            uv=np.concatenate([uv,uv]);joint=np.concatenate([joint,joint])
            weight=np.concatenate([weight,weight]);color=np.concatenate([color,color])
            triangles=np.concatenate([triangles,triangles[:,[0,2,1]]+n]);n*=2
        pos.append(position);norm.append(normal);uvs.append(uv);joints.append(joint);weights.append(weight);colors.append(color)
        indices.append(triangles.reshape(-1)+base);base+=n
        r = int(round(pbr.get('roughnessFactor', .8)*255))
        rough.paste((255,r,0),(x-8,y-8,x+w+8,y+h+8))
    rough = limit_image(rough, max_side)
    binary=bytearray(raw[:g['buffers'][0]['byteLength']])
    attributes={kind:append_accessor(g,binary,values,typ,component) for kind,values,typ,component in [
        ('POSITION',np.concatenate(pos),'VEC3',5126),('NORMAL',np.concatenate(norm),'VEC3',5126),
        ('TEXCOORD_0',np.concatenate(uvs),'VEC2',5126),('JOINTS_0',np.concatenate(joints),'VEC4',5123),
        ('WEIGHTS_0',np.concatenate(weights),'VEC4',5126),('COLOR_0',np.concatenate(colors),'VEC4',5126)]}
    index=append_accessor(g,binary,np.concatenate(indices).reshape(-1,1),'SCALAR',5125,34963)
    g['meshes']=[{'name':source_path.stem+'_WorkshopSingleMesh','primitives':[{'attributes':attributes,'indices':index,'material':0,'mode':4}]}]
    g['images']=[];g['textures']=[]
    for name,im in [('atlas',atlas),('roughness',rough)]:
        stream=io.BytesIO();im.save(stream,format='PNG');png=stream.getvalue();binary+=b'\0'*(-len(binary)%4)
        view=len(g['bufferViews']);g['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':len(png)});binary.extend(png)
        g['images'].append({'name':name,'mimeType':'image/png','bufferView':view});g['textures'].append({'source':len(g['images'])-1,'sampler':0})
    g['samplers']=[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071}]
    g['materials']=[{'name':target_path.stem,'doubleSided':False,'alphaMode':'MASK','alphaCutoff':.3333,
                     'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicRoughnessTexture':{'index':1},'metallicFactor':0,'roughnessFactor':1}}]
    g['buffers']=[{'byteLength':len(binary)}]
    g.setdefault('extras',{})['workshopCompatibility']='One mesh, one primitive, one material; embedded atlas and roughness, original skin retained.'
    g['extras']['sidednessMode']='Front-facing skin; two-sided source surfaces have explicit reversed faces and flipped normals.'
    target_path.parent.mkdir(parents=True,exist_ok=True);write_glb(target_path,g,bytes(binary))
    report={'file':str(target_path),'meshes':1,'primitives':1,'materials':1,'embeddedImages':2,
            'vertices':base,'triangles':sum(len(i)//3 for i in indices),'atlasSize':atlas.size,
            'sourceAtlasSize':source_atlas_size,'nativeTextureMaxSide':max_side,
            'sourceSha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'glbSha256':hashlib.sha256(target_path.read_bytes()).hexdigest(),
            'repeatedTextureBakes':repeat_bakes,
            'sidednessBake':{'originalVertices':original_vertices,'originalTriangles':original_triangles,
                'addedVertices':base-original_vertices,'addedTriangles':sum(len(i)//3 for i in indices)-original_triangles,
                'parts':sided_parts,'finalMaterialDoubleSided':False}}
    return report
