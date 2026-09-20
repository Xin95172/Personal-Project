"""Pinned project-local PortAudio baseline/fixed variants; never installs a DLL."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = '147dd722548358763a8b649b3e4b41dfffbcfbb6'


def patched(source):
    old = 'UINT32 bufferFrames = ALIGN_NEXT_POW2((stream->in.framesPerHostCallback / WASAPI_PACKETS_PER_INPUT_BUFFER) * 2);'
    new = '''/* A polling capture packet can fill the whole endpoint buffer.
             * ReadStream drains existing tail before fetching another packet,
             * so one full host buffer bounds every possible unconsumed suffix. */
            UINT32 bufferFrames = ALIGN_NEXT_POW2(stream->in.framesPerHostCallback);'''
    write = 'PaUtil_WriteRingBuffer(stream->in.tailBuffer, wasapi_buffer + bytes_processed, frames_to_save);'
    checked = '''/* Never silently release frames that did not fit the tail. */
            if ((UINT32)PaUtil_WriteRingBuffer(stream->in.tailBuffer, wasapi_buffer + bytes_processed, frames_to_save) != frames_to_save)
            {
                hr = IAudioCaptureClient_ReleaseBuffer(stream->captureClient, available);
                SetEvent(stream->hBlockingOpStreamRD);
                return (hr != S_OK ? paUnanticipatedHostError : paInputOverflowed);
            }'''
    if source.count(old) != 1 or source.count(write) != 1:
        raise ValueError('Pinned PortAudio patch context mismatch')
    return source.replace(old, new).replace(write, checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['baseline', 'fixed'], required=True)
    args = parser.parse_args()
    upstream = ROOT / 'out/portaudio-upstream'
    if not upstream.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v19.7.0',
                        'https://github.com/PortAudio/portaudio.git', str(upstream)], check=True)
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Unexpected PortAudio revision')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Preserve edits in PortAudio upstream before generation')
    destination = ROOT / f'out/portaudio-{args.variant}-src'
    manifest_path = destination / '.virmixer-portaudio.json'
    if destination.exists():
        if not manifest_path.exists():
            raise ValueError('Unmanaged existing source directory')
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest['files'].items():
            if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Generated source changed; preserve edits: ' + relative)
    shutil.copytree(upstream, destination, ignore=shutil.ignore_patterns('.git'), dirs_exist_ok=True)
    source = destination / 'src/hostapi/wasapi/pa_win_wasapi.c'
    if args.variant == 'fixed':
        source.write_text(patched(source.read_text()), encoding='utf-8')
    files = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in destination.rglob('*') if p.is_file() and p != manifest_path}
    manifest_path.write_text(json.dumps(dict(revision=revision, variant=args.variant, files=files), indent=2))
    print(destination)


if __name__ == '__main__':
    main()
