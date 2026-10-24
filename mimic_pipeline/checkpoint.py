"""Single implementation of checkpoint integrity and upstream-version inspection."""
from pathlib import Path
import pickletools
import shutil
import subprocess
import zipfile
from .common import sha256


def git_commit(repository):
    repository=Path(repository)
    executable=shutil.which('git')
    if executable:
        result=subprocess.run([executable,'-C',str(repository),'rev-parse','HEAD'],capture_output=True,text=True)
        if result.returncode==0:return result.stdout.strip()
    git_dir=repository/'.git';head=git_dir/'HEAD'
    if head.is_file():
        reference=head.read_text(encoding='utf-8').strip()
        if not reference.startswith('ref: '):return reference
        reference=reference[5:]
        if (git_dir/reference).is_file():return (git_dir/reference).read_text(encoding='utf-8').strip()
        packed=git_dir/'packed-refs'
        if packed.is_file():
            for line in packed.read_text(encoding='utf-8').splitlines():
                if line.endswith(' '+reference):return line.split()[0]
    return None


def inspect_checkpoint(checkpoint,repository):
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        bad=archive.testzip()
        if bad:raise ValueError('Checkpoint CRC failed: '+bad)
        metadata=next(name for name in archive.namelist() if name.endswith('/data.pkl') or name=='data.pkl')
        strings=[value for operation,value,_ in pickletools.genops(archive.read(metadata))
                 if operation.name in ('BINUNICODE','SHORT_BINUNICODE','UNICODE') and isinstance(value,str)]
    if not any(x.startswith('vision_model.model.encoder.layers.') for x in strings) or not any(x.startswith('text_model.model.embeddings.') for x in strings):
        raise ValueError('Expected the complete Swin vision and text encoders')
    return {'checkpoint':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':sha256(checkpoint),
            'zip_crc_passed':True,'swin_and_text_keys_present':True,'medclip_commit':git_commit(repository),
            'status':'checkpoint_inspected_only','actual_output_dimension':None,'forward_pass_verified':False}
