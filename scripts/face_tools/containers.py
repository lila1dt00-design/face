"""Data-only extraction of archives, CPK/FPK and referenced FTEX textures."""
import io,json,os,re,subprocess,zipfile
from pathlib import Path,PurePosixPath,PureWindowsPath
from PIL import Image
from .paths import normalize_texture
from .third_party.pes_file_tools.cpk import CpkReader
from .third_party.pes_file_tools.fpk import FpkFile
from .third_party.pes_file_tools.FmdlFile import FmdlFile
from .third_party.pes_file_tools.ftex import ftexToDdsBuffer

BLOCKED_EXTENSIONS={'.exe','.bat','.cmd','.ps1','.vbs','.js','.scr','.msi','.dll','.com','.lnk','.url'}

def decode_tar_lines(raw):
    lines=[]
    for line in raw.splitlines():
        for enc in ('utf-8','gb18030','cp1252'):
            try:
                lines.append(line.decode(enc));break
            except UnicodeDecodeError:pass
        else:lines.append(line.decode('utf-8',errors='replace'))
    return lines

def safe_target(root,member):
    name=member.replace('\\','/');path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or PureWindowsPath(name).drive:raise ValueError('Unsafe archive path: '+member)
    target=(Path(root)/path).resolve();target.relative_to(Path(root).resolve());return target

def _unpack_7z(source,target,member_prefix,max_bytes):
    """Extract a .7z with py7zr (pure Python, works on any PC). Returns (count,bytes) or None if py7zr is unavailable."""
    try:import py7zr
    except Exception:return None
    try:
        with py7zr.SevenZipFile(source,'r') as z:
            infos=[i for i in z.list() if not i.is_directory and not (member_prefix and not i.filename.replace('\\','/').startswith(member_prefix)) and Path(i.filename).suffix.lower() not in BLOCKED_EXTENSIONS]
            total=sum(i.uncompressed or 0 for i in infos)
            if total>max_bytes:raise ValueError('Expanded archive exceeds bound; select a narrower member prefix.')
            names=[i.filename for i in infos]
            if not names:return 0,0
            data=z.read(targets=names)
        for name,stream in data.items():
            dest=safe_target(target,name);dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as out:out.write(stream.read())
        return len(data),total
    except ValueError:raise
    except Exception:
        # Unsupported method or py7zr problem: remove partial output and let the tar fallback try.
        import shutil
        for child in Path(target).iterdir():
            shutil.rmtree(child,ignore_errors=True) if child.is_dir() else child.unlink()
        return None

def unpack_archive(source,target,member_prefix=None,max_bytes=2*1024**3):
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=False);total=0;count=0
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            for entry in z.infolist():
                if entry.is_dir() or (member_prefix and not entry.filename.startswith(member_prefix)):continue
                if Path(entry.filename).suffix.lower() in BLOCKED_EXTENSIONS:continue
                dest=safe_target(target,entry.filename);total+=entry.file_size
                if total>max_bytes:raise ValueError('Expanded archive size exceeds bound; select a narrower member prefix.')
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(entry) as inp,dest.open('xb') as out:
                    import shutil
                    shutil.copyfileobj(inp,out)
                count+=1
    elif source.suffix.lower()=='.7z' and (res:=_unpack_7z(source,target,member_prefix,max_bytes)) is not None:
        count,total=res
    else:
        # bsdtar can read the retained RAR/7z packs on Windows. It emits each file
        # to stdout; symlinks and archive paths never create filesystem links.
        result=subprocess.run(['tar','-tf',str(source)],capture_output=True,timeout=60,check=True)
        names=decode_tar_lines(result.stdout)
        dirs=set()
        for name in names:
            parts=name.rstrip('/').split('/')
            dirs.update('/'.join(parts[:i]) for i in range(1,len(parts)))
        for member in names:
            if member.endswith('/') or member.rstrip('/') in dirs or (member_prefix and not member.startswith(member_prefix)):continue
            if Path(member).suffix.lower() in BLOCKED_EXTENSIONS:continue
            dest=safe_target(target,member)
            # libarchive's matcher uses backslash escapes, not fnmatch bracket literals.
            pattern=''.join('\\'+c if c in '[]*?' else c for c in member)
            result=subprocess.run(['tar','-xOf',str(source),'--',pattern],capture_output=True,timeout=90,check=True)
            total+=len(result.stdout)
            if total>max_bytes:raise ValueError('Expanded archive exceeds bound; select a narrower member prefix.')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as out:out.write(result.stdout)
            count+=1
    if not count:raise ValueError('Archive selection is empty.')
    return dict(files=count,bytes=total)

def cpk_ids(source):
    cpk=CpkReader();cpk.open(str(source))
    try:return sorted(set(m[1] for entry in cpk.files if (m:=re.search(r'/face/real/([^/]+)/','/'+entry.name.replace('\\','/'),re.I))))
    finally:
        if hasattr(cpk,'close'):cpk.close()

def extract_cpk(source,face_id,target):
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=True);cpk=CpkReader();cpk.open(str(source));count=0
    try:
        entries={normalize_texture('/'+e.name.lstrip('/')):e for e in cpk.files}
        marker='/face/real/'+str(face_id).lower()+'/'
        for key,entry in entries.items():
            if marker not in key:continue
            dest=safe_target(target,entry.name.lstrip('/'));dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as out:out.write(cpk.readFile(entry))
            count+=1
        if not count:raise ValueError('Requested face ID is not present in this CPK.')
        # Resolve FMDL references that point to a different donor texture folder.
        for fpk in list(target.rglob('face.fpk')):unpack_fpk(fpk,fpk.parent)
        for fmdl in list(target.rglob('*_high.fmdl')):
            model=FmdlFile();model.readFile(fmdl)
            for mesh in model.meshes:
                for role,tex in mesh.materialInstance.textures:
                    key=normalize_texture(tex.directory.rstrip('/')+'/'+tex.filename).rsplit('.',1)[0]+'.ftex'
                    if key not in entries:continue
                    entry=entries[key];dest=safe_target(target,entry.name.lstrip('/'))
                    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(cpk.readFile(entry));count+=1
        return count
    finally:
        if hasattr(cpk,'close'):cpk.close()

def unpack_fpk(source,target):
    fpk=FpkFile();fpk.read(Path(source).read_bytes());target=Path(target);target.mkdir(parents=True,exist_ok=True)
    for name,raw in fpk.entries.items():
        dest=safe_target(target,Path(name.replace('\\','/')).name)
        if dest.exists():
            if dest.read_bytes()!=raw:raise ValueError('FPK basename collision: '+name)
        else:dest.write_bytes(raw)

def resolve_textures(folder,search_root):
    folder=Path(folder);search_root=Path(search_root);index={};files=[p for p in search_root.rglob('*') if p.is_file()]
    skipped=[]
    old_index=json.loads((folder/'texture-index.json').read_text(encoding='utf8')) if (folder/'texture-index.json').exists() else {}
    def pick(tex,key):
        token=re.search(r'/face/real/([^/]+)/',key)
        candidates=[p for p in files if p.stem.lower()==Path(tex.filename).stem.lower() and p.suffix.lower() in ['.png','.ftex','.dds']]
        tail=key.split('/face/real/',1)[-1].rsplit('.',1)[0]
        exact_path=[p for p in candidates if normalize_texture(str(p.relative_to(search_root)).replace('\\','/')).rsplit('.',1)[0].endswith(tail)]
        if exact_path:candidates=exact_path
        if key in old_index:
            original=(folder/old_index[key]).resolve();original.relative_to(search_root.resolve())
            same=[p for p in candidates if p.parent==original.parent]
            if same:candidates=same
        if token:
            exact=[p for p in candidates if '/'+token[1]+'/' in '/'+str(p.relative_to(search_root)).replace('\\','/').lower()]
            if exact:candidates=exact
        by_parent={}
        for p in candidates:by_parent.setdefault(p.parent,[]).append(p)
        if len(by_parent)!=1:raise ValueError('Missing or ambiguous original base texture: '+key)
        options=next(iter(by_parent.values()));selected=next((p for p in options if p.suffix.lower()=='.png'),None)
        if selected is None:
            source=next((p for p in options if p.suffix.lower()=='.ftex'),options[0]);raw=source.read_bytes()
            if source.suffix.lower()=='.ftex':raw=ftexToDdsBuffer(raw)
            elif raw[4:8]==b'ESYS':
                import zlib
                raw=zlib.decompress(raw[16:])
            selected=source.with_suffix('.png')
            with Image.open(io.BytesIO(raw)) as im:im.convert('RGBA').save(selected)
        textures=folder/'resolved_textures';textures.mkdir(exist_ok=True)
        dest=textures/(str(len(index))+'_'+selected.name);dest.write_bytes(selected.read_bytes())
        return str(dest.relative_to(folder).with_suffix('.ftex')).replace('\\','/')
    for filename in ['face_high.fmdl','hair_high.fmdl']:
        if not (folder/filename).exists():continue
        model=FmdlFile();model.readFile(folder/filename)
        for mesh in model.meshes:
            base_name=re.sub(r'\.\d+$','',mesh.materialInstance.name)
            if base_name not in ['fox_skin_mat','fox_hair_mat','fox_head_shell_mat','fox_hair_transfer_mat','fox_hair_decal_mat']:continue
            for role,tex in mesh.materialInstance.textures:
                if role!='Base_Tex_SRGB':continue
                key=normalize_texture(tex.directory.rstrip('/')+'/'+tex.filename).rsplit('.',1)[0]+'.ftex'
                if key in index:continue
                try:
                    index[key]=pick(tex,key)
                except ValueError:
                    # Opt-in fallback: the hair-transfer coefficient map is absent from some packs.
                    # Use the same material's real colour strand map (Base_Tex_2) instead.
                    alt=next((t for r,t in mesh.materialInstance.textures if r=='Base_Tex_2_SRGB'),None)
                    if base_name=='fox_hair_transfer_mat' and alt is not None and os.environ.get('FACE_SKIP_MISSING_TRANSFER')=='1':
                        altkey=normalize_texture(alt.directory.rstrip('/')+'/'+alt.filename).rsplit('.',1)[0]+'.ftex'
                        index[key]=pick(alt,altkey);skipped.append(key+' -> substituted '+alt.filename)
                    else:raise
    (folder/'texture-index.json').write_text(json.dumps(index,indent=2),encoding='utf8')
    if skipped:(folder/'skipped-textures.json').write_text(json.dumps(skipped),encoding='utf8')
    return index
