import unittest
from parse_cable_diagnostics import parse, summarize


LINE = (
    "VirMixer: UNDERRUN reset=4 count=9 req=192 avail=0 take=0 primed=1 "
    "totalW=12000 totalReq=12192 totalRead=12000 epochW=2400 epochReq=2592 epochRead=2400"
)


class CableDiagnosticsTests(unittest.TestCase):
    def test_classifies_temporary_phase_deficit(self):
        result = summarize(parse(LINE))
        self.assertEqual(result["underruns"], 1)
        self.assertEqual(result["empty_reads"], 1)
        self.assertEqual(result["impossible_actual_read_ahead_of_written"], 0)
        self.assertFalse(result["possible_reset_race"])

    def test_identifies_empty_epoch(self):
        line = LINE.replace("epochW=2400", "epochW=0").replace("epochRead=2400", "epochRead=0")
        result = summarize(parse(line))
        self.assertTrue(result["possible_reset_race"])

    def test_ignores_unrelated_lines(self):
        self.assertEqual(summarize(parse("other driver output"))["underruns"], 0)


if __name__ == "__main__":
    unittest.main()
