import unittest
from wasapi_capture import PacketTrace


class PacketTraceTests(unittest.TestCase):
    def row(self, position, flags=0):
        return dict(device_position=position, frames=480, flags=flags)

    def test_startup_and_gap(self):
        trace = PacketTrace()
        trace.add(self.row(0, 1))
        trace.add(self.row(480))
        self.assertIsNone(trace.trigger)
        trace.add(self.row(1360, 1))
        self.assertEqual(trace.rows[-1]['position_gap'], 400)
        self.assertEqual(trace.trigger, 3)
        for i in range(64):
            trace.add(self.row(1840 + i * 480))
        self.assertTrue(trace.frozen)
        frozen = list(trace.rows)
        trace.add(self.row(999999, 4))
        self.assertEqual(list(trace.rows), frozen)
        self.assertEqual(trace.timestamp_errors, 1)

    def test_bounded_without_trigger(self):
        trace = PacketTrace()
        for i in range(1000):
            trace.add(self.row(i * 480))
        self.assertEqual(len(trace.rows), 256)
        self.assertIsNone(trace.trigger)

    def test_source_jump_with_continuous_device_positions(self):
        trace = PacketTrace()
        row = self.row(0)
        row.update(first_word=0xa0000000, last_word=0xa0000000 | 479)
        trace.add(row)
        row = self.row(480)
        row.update(first_word=0xa0000000 | 880, last_word=0xa0000000 | 1359)
        trace.add(row)
        self.assertEqual(trace.trigger, 2)
        self.assertEqual(trace.rows[-1]['position_gap'], 0)
        self.assertTrue(trace.rows[-1]['source_discontinuity'])


if __name__ == '__main__':
    unittest.main()
