import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from mixer.device_panel import DevicePanel
from mixer.devices import refresh_and_resolve


def endpoint(name, inputs, outputs, index=0):
    return dict(name=name, host='WASAPI', inputs=inputs, outputs=outputs,
                index=index, sample_rate=48000, default_input=False, default_output=False)


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_inputs_outputs_search_and_selection(self):
        panel = DevicePanel()
        panel.set_inventory([endpoint('Microphone', 1, 0), endpoint('Speaker', 0, 2), endpoint('Duplex', 2, 2)])
        self.assertEqual(panel.table.rowCount(), 3)
        panel.direction.setCurrentIndex(1)
        self.assertEqual(panel.table.rowCount(), 2)
        panel.table.selectRow(0)
        self.assertFalse(panel.use.isEnabled())
        panel.direction.setCurrentIndex(2)
        self.assertEqual(panel.table.rowCount(), 2)
        panel.search.setText('speaker')
        self.assertEqual(panel.table.rowCount(), 1)
        panel.table.selectRow(0)
        self.assertTrue(panel.use.isEnabled())
        panel.output_active = True
        panel.selection_changed()
        self.assertFalse(panel.use.isEnabled())
        panel.close()

    def test_resolve_after_index_change_and_removal(self):
        selection = endpoint('Speaker', 0, 2, index=3)
        with patch('mixer.devices.sd._terminate'), patch('mixer.devices.sd._initialize'), \
                patch('mixer.devices.inventory', return_value=[endpoint('Speaker', 0, 2, index=12)]):
            self.assertEqual(refresh_and_resolve(selection), 12)
        with patch('mixer.devices.sd._terminate'), patch('mixer.devices.sd._initialize'), \
                patch('mixer.devices.inventory', return_value=[]):
            with self.assertRaises(ValueError):
                refresh_and_resolve(selection)

    def test_ambiguous_device_never_routes_to_arbitrary_endpoint(self):
        duplicate = [endpoint('USB Audio', 0, 2, i) for i in (1, 2)]
        with patch('mixer.devices.sd._terminate'), patch('mixer.devices.sd._initialize'), \
                patch('mixer.devices.inventory', return_value=duplicate):
            with self.assertRaises(ValueError):
                refresh_and_resolve(duplicate[0])


if __name__ == '__main__':
    unittest.main()
