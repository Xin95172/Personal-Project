"""Regeneration must be reproducible and must protect local generated-source edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PrepareTests(unittest.TestCase):
    def test_frame_probe_is_explicit_debug_b_only(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='virmixer-probe-') as temp:
            destination = Path(temp) / 'generated'
            command = [sys.executable, str(root / 'prepare.py'), '--output-dir', str(destination), '--frame-probe']
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            result = subprocess.run(command + ['--shared-timeline'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for relative in ('EndpointsCommon/EndpointsCommon.vcxproj', 'TabletAudioSample/TabletAudioSample.vcxproj'):
                content = (destination / 'audio/sysvad' / relative).read_text()
                self.assertEqual(content.count('VIRMIXER_FRAME_PROBE=1'), 1)
            stream = (destination / 'audio/sysvad/EndpointsCommon/minwavertstream.cpp').read_text()
            update = stream[stream.index('VOID CMiniportWaveRTStream::UpdatePosition'):]
            self.assertLess(update.index('->ProbeDma(false,'), update.index('ReadBytes(ByteDisplacement'))
            self.assertGreater(update.index('->ProbeDma(true,'), update.index('WriteBytes(ByteDisplacement'))
            self.assertLess(update.index('->ProbeDma(true,'), update.index('m_ullLinearPosition += ByteDisplacement'))
            self.assertIn('!VIRMIXER_FRAME_PROBE && m_ulNotificationsPerBuffer > 0', update)

    def test_explicit_ab_generation_mode(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='virmixer-ab-') as temp:
            destination = Path(temp) / 'generated'
            command = [sys.executable, str(root / 'prepare.py'), '--output-dir', str(destination)]
            for mode, options in [('A', []), ('B', ['--shared-timeline']), ('A', [])]:
                result = subprocess.run(command + options, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((destination / '.virmixer-mode.json').read_text())['mode'], mode)
                for relative in ('EndpointsCommon/EndpointsCommon.vcxproj','TabletAudioSample/TabletAudioSample.vcxproj'):
                    text = (destination / 'audio/sysvad' / relative).read_text()
                    self.assertIn('VIRMIXER_SHARED_TIMELINE=' + str(int(mode == 'B')), text)
                    self.assertNotIn('VIRMIXER_SHARED_TIMELINE=' + str(int(mode != 'B')), text)

    def test_repeat_generation_and_local_edit_protection(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='virmixer-generation-') as temp:
            destination = Path(temp) / 'generated'
            command = [sys.executable, str(root / 'prepare.py'), '--output-dir', str(destination)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = (destination / '.virmixer-generated.json').read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(before, (destination / '.virmixer-generated.json').read_bytes())
            hashes = json.loads(before)
            for name, digest in hashes.items():
                self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), digest)
            edited = destination / 'audio/sysvad/common.cpp'
            content = edited.read_bytes() + b'\n// local edit must survive\n'
            edited.write_bytes(content)
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('Generated files were changed', rejected.stderr)
            self.assertEqual(edited.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
