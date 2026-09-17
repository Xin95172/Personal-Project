"""Offline package consistency checks; no installation or trust-store writes."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def verify(package):
    manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8-sig'))
    files = {name.lower(): digest for name, digest in manifest['files'].items()}
    assert set(files) == {'virmixeraudio.sys', 'virmixeraudio.inf', 'virmixeraudio.cat'}, files
    for name, digest in files.items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest().upper() == digest.upper(), name
    raw = (package / 'VirMixerAudio.inf').read_bytes()
    inf = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
    for expected in ['Root\\VirMixerAudio', 'VirMixer Input', 'VirMixer Output',
                     'ServiceBinary=%13%\\VirMixerAudio.sys', 'CatalogFile=VirMixerAudio.cat',
                     'WaveSpdif', 'WaveMicIn', 'TopologySpdif', 'TopologyMicIn']:
        assert expected.lower() in inf.lower(), expected
    assert '$KMDFVERSION$' not in inf
    assert 'TabletAudioSample.sys' not in inf
    binary = (package / 'VirMixerAudio.sys').read_bytes()
    assert binary[:2] == b'MZ'
    pe = struct.unpack_from('<I', binary, 0x3c)[0]
    assert binary[pe:pe+4] == b'PE\0\0'
    assert struct.unpack_from('<H', binary, pe+4)[0] == 0x8664, 'Expected x64 binary'
    assert (package / 'VirMixerAudio.cat').stat().st_size > 0
    print(f'PASS: package hashes, x64 SYS, service, hardware ID and endpoint INF: {package}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    verify(parser.parse_args().package)
