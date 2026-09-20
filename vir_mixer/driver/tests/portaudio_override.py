"""Load one explicitly chosen DLL for this process before importing sounddevice."""
import ctypes.util
import importlib
from pathlib import Path
import sys


def load(path):
    if 'sounddevice' in sys.modules:
        raise RuntimeError('Select the diagnostic DLL before importing sounddevice')
    target = Path(path).resolve(strict=True)
    original = ctypes.util.find_library
    try:
        ctypes.util.find_library = lambda name: str(target) if name == 'portaudio' else original(name)
        sd = importlib.import_module('sounddevice')
    finally:
        ctypes.util.find_library = original
    if Path(sd._libname).resolve() != target:
        raise RuntimeError('sounddevice did not load the requested DLL; refusing silent fallback')
    return sd


if __name__ == '__main__':
    import runpy
    if len(sys.argv) < 3:
        raise SystemExit('Usage: portaudio_override.py DLL TEST_SCRIPT [arguments...]')
    load(sys.argv[1])
    script = Path(sys.argv[2]).resolve(strict=True)
    sys.argv = [str(script), *sys.argv[3:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')
