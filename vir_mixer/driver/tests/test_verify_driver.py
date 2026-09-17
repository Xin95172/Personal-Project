import unittest
import numpy as np
from verify_driver import analyze

class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.ref = np.random.default_rng(97).integers(-3000, 3001, (4800, 2), dtype=np.int16)
    def test_correct_offset(self):
        data = np.pad(self.ref, ((973, 800), (0, 0)))
        self.assertEqual(analyze(data, self.ref)['offset_frames'], [973, 973])
    def test_reject_bad_pcm(self):
        for bad in [np.zeros_like(self.ref), self.ref[:, ::-1],
                    np.repeat(self.ref[:, :1], 2, axis=1), -self.ref,
                    self.ref // 2, np.roll(self.ref, 200, axis=0)]:
            with self.subTest(), self.assertRaises(AssertionError):
                analyze(bad, self.ref)
    def test_reject_timing_mismatch(self):
        data = np.pad(self.ref, ((50, 50), (0, 0)))
        data[:, 1] = np.roll(data[:, 1], 3)
        with self.assertRaises(AssertionError): analyze(data, self.ref)
    def test_reject_short_capture(self):
        with self.assertRaises(AssertionError): analyze(self.ref[:100], self.ref)

if __name__ == '__main__': unittest.main()
