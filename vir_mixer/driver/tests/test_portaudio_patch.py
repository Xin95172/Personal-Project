"""Guard the pinned client-side patch and its fail-closed source context."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'portaudio_prepare', Path(__file__).resolve().parents[1] / 'portaudio/prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PortAudioPatchTests(unittest.TestCase):
    def test_changed_upstream_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare.patched('unrecognized upstream')

    def test_pinned_source_patch(self):
        source = prepare.ROOT / 'out/portaudio-upstream/src/hostapi/wasapi/pa_win_wasapi.c'
        if not source.exists():
            self.skipTest('Run driver/portaudio/prepare.py to fetch pinned source')
        original = source.read_text()
        patched = prepare.patched(original)
        self.assertIn('bufferFrames = ALIGN_NEXT_POW2(stream->in.framesPerHostCallback);', patched)
        self.assertIn('!= frames_to_save)', patched)
        self.assertIn('paUnanticipatedHostError : paInputOverflowed', patched)
        # One additional ReleaseBuffer only on the immediate-return error path.
        self.assertEqual(patched.count('IAudioCaptureClient_ReleaseBuffer('),
                         original.count('IAudioCaptureClient_ReleaseBuffer(') + 1)
        with self.assertRaises(ValueError):
            prepare.patched(patched)


if __name__ == '__main__':
    unittest.main()
