import tempfile
from pathlib import Path
import unittest
import numpy as np
from frame_identity import signal, analyze_capture, analyze_log


class IdentityTests(unittest.TestCase):
    def test_strict_identity_even_when_transport_reports_no_errors(self):
        cases = [
            (np.concatenate((signal(0, 100), signal(148, 100))), 248, False, 48),
            (np.concatenate((signal(0, 100), signal(50, 100))), 200, False, 0),
            (signal(0, 199), 200, False, 0),
            (np.concatenate((np.zeros((20, 2), dtype=np.int16), signal(0, 200))), 200, True, 0),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'capture.raw'
            for data, expected, passed, gaps in cases:
                path.write_bytes(data.tobytes())
                result = analyze_capture(path, expected)
                self.assertEqual(result['pcm_passed'], passed)
                self.assertEqual(result['source_gap_frames'], gaps)

    def test_structured_loss(self):
        data = np.concatenate((signal(0, 100), signal(148, 672), signal(1172, 100)))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'capture.raw'
            path.write_bytes(data.tobytes())
            result = analyze_capture(path)
            self.assertEqual([x['offset'] for x in result['transitions']], [0, 48, 400])
            self.assertEqual([x['delta'] for x in result['transitions']], [None, 48, 352])

    def test_silence_hole_is_not_shift(self):
        data = np.concatenate((signal(0, 100), np.zeros((176, 2), dtype=np.int16), signal(276, 100)))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'capture.raw'
            path.write_bytes(data.tobytes())
            result = analyze_capture(path)
            self.assertEqual(len(result['transitions']), 1)
            self.assertEqual(result['zero_frames'], 176)

    def test_log_requires_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'debug.log'
            path.write_text('VirMixer: FRAME seq=1 epoch=2 stage=0 frame=3 word=2684354563 offset=0 reason=0\n', encoding='utf-16')
            self.assertFalse(analyze_log(path)['complete'])
            with path.open('a', encoding='utf-16') as output:
                output.write('VirMixer: FRAME_END seq=1 trigger=0 frozen=0\n')
            self.assertTrue(analyze_log(path)['complete'])
            path.write_text('VirMixer: FRAME seq=2 reason=0\nVirMixer: FRAME_END seq=2 trigger=0 frozen=0\n')
            self.assertFalse(analyze_log(path)['complete'])


if __name__ == '__main__':
    unittest.main()
